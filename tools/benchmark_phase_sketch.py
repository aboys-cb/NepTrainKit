#!/usr/bin/env python
"""Benchmark the experimental PhaseSketch against OVITO PTM.

The benchmark is synthetic by design: training and test structures are
generated from separate distortion seeds, and the test split contains stronger
strain/noise, vacancies, skewed cells, and structures absent from the reference
bank.  It measures structure-level phase identification, which is the first
decision required by the Training Set Audit phase atlas.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.build import bulk
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import ExtraTreesClassifier

from NepTrainKit.core.audit.phase_sketch import (
    PrototypeBank,
    phase_sketch,
    summarize_phase_sketch,
)


warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
phase_sketch_module = importlib.import_module("NepTrainKit.core.audit.phase_sketch")


_MINIMUM_LOCAL_GEOMETRY_SUPPORT = 0.40
_LOCAL_EVIDENCE_FOREST_PROBABILITY = 0.80


@dataclass(frozen=True)
class Frame:
    atoms: Atoms
    geometry: str
    ordering: str
    condition: str
    source: str

    @property
    def phase(self) -> str:
        if self.geometry == "fcc":
            return f"fcc:{self.ordering}"
        if self.geometry == "bcc":
            return f"bcc:{self.ordering}"
        return self.geometry


@dataclass(frozen=True)
class GeometryEvidence:
    """Structure label plus the evidence needed to expose weak decisions."""

    label: str
    candidate: str
    confidence_state: str
    local_support_fraction: float
    local_unknown_fraction: float
    local_margin_median: float
    local_distance_median: float
    local_evidence_evaluated: bool
    structure_distance_ratio: float
    forest_probability: float
    translational_order_score: float | None
    translational_order_limit: float | None
    cna_phase_fractions: dict[str, float]


def _random_solution_symbols(
    atom_count: int,
    symbols: tuple[str, ...],
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.full(len(symbols), 1.5))
    decoration = rng.choice(np.asarray(symbols, dtype=object), atom_count, p=probabilities)
    decoration[rng.choice(atom_count, len(symbols), replace=False)] = symbols
    rng.shuffle(decoration)
    return decoration.tolist()


def _fcc_a1(seed: int, species_count: int = 2, varied_composition: bool = False) -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((3, 3, 3))
    pool = ("Cu", "Au", "Ag", "Pd")[:species_count]
    if varied_composition or species_count != 2:
        atoms.set_chemical_symbols(_random_solution_symbols(len(atoms), pool, seed))
        return atoms
    rng = np.random.default_rng(seed)
    symbols = np.asarray(["Cu"] * len(atoms), dtype=object)
    symbols[rng.choice(len(atoms), len(atoms) // 2, replace=False)] = "Au"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def _bcc_a2(seed: int, species_count: int = 2, varied_composition: bool = False) -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.88, cubic=True).repeat((4, 4, 4))
    pool = ("Fe", "Al", "Cr", "V")[:species_count]
    if varied_composition or species_count != 2:
        atoms.set_chemical_symbols(_random_solution_symbols(len(atoms), pool, seed))
        return atoms
    rng = np.random.default_rng(seed)
    symbols = np.asarray(["Fe"] * len(atoms), dtype=object)
    symbols[rng.choice(len(atoms), len(atoms) // 2, replace=False)] = "Al"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def _l10() -> Atoms:
    a, c = 3.82, 3.70
    scaled = np.asarray(
        ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    )
    atoms = Atoms(
        symbols=("Au", "Au", "Cu", "Cu"),
        scaled_positions=scaled,
        cell=np.diag((a, a, c)),
        pbc=True,
    )
    return atoms.repeat((3, 3, 3))


def _l12() -> Atoms:
    a = 3.75
    scaled = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
    )
    atoms = Atoms(
        symbols=("Au", "Cu", "Cu", "Cu"),
        scaled_positions=scaled,
        cell=np.diag((a, a, a)),
        pbc=True,
    )
    return atoms.repeat((3, 3, 3))


def _l11() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.72, cubic=True).repeat((4, 4, 4))
    half_grid = np.rint(atoms.get_scaled_positions() * np.asarray((4, 4, 4)) * 2).astype(int)
    layers = (np.sum(half_grid, axis=1) // 2) % 2
    atoms.set_chemical_symbols(np.where(layers == 0, "Cu", "Au").tolist())
    return atoms


def _b2() -> Atoms:
    atoms = Atoms(
        symbols=("Fe", "Al"),
        scaled_positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        cell=np.diag((2.92, 2.92, 2.92)),
        pbc=True,
    )
    return atoms.repeat((4, 4, 4))


def _a15() -> Atoms:
    scaled = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.5, 0.25),
            (0.0, 0.5, 0.75),
            (0.5, 0.25, 0.0),
            (0.5, 0.75, 0.0),
            (0.25, 0.0, 0.5),
            (0.75, 0.0, 0.5),
        )
    )
    atoms = Atoms(
        symbols=("Si", "Si", "Cr", "Cr", "Cr", "Cr", "Cr", "Cr"),
        scaled_positions=scaled,
        cell=np.diag((4.56, 4.56, 4.56)),
        pbc=True,
    )
    return atoms.repeat((3, 3, 3))


def _amorphous(seed: int, atom_count: int = 128) -> Atoms:
    rng = np.random.default_rng(seed)
    side = 13.0
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < atom_count and attempts < atom_count * 5000:
        attempts += 1
        candidate = rng.uniform(0.0, side, size=3)
        if not positions:
            positions.append(candidate)
            continue
        current = np.asarray(positions)
        delta = current - candidate
        delta -= side * np.rint(delta / side)
        if float(np.min(np.linalg.norm(delta, axis=1))) > 1.65:
            positions.append(candidate)
    if len(positions) != atom_count:
        raise RuntimeError("failed to generate amorphous benchmark structure")
    symbols = np.where(np.arange(atom_count) % 2, "Cu", "Zr").tolist()
    return Atoms(symbols=symbols, positions=np.asarray(positions), cell=np.eye(3) * side, pbc=True)


def _base_structures(seed: int) -> dict[str, tuple[Atoms, str, str]]:
    return {
        "fcc_pure": (bulk("Cu", "fcc", a=3.62, cubic=True).repeat((3, 3, 3)), "fcc", "pure"),
        "fcc_a1": (_fcc_a1(seed), "fcc", "a1"),
        "fcc_l10": (_l10(), "fcc", "l10"),
        "fcc_l12": (_l12(), "fcc", "l12"),
        "bcc_pure": (bulk("Fe", "bcc", a=2.88, cubic=True).repeat((4, 4, 4)), "bcc", "pure"),
        "bcc_a2": (_bcc_a2(seed), "bcc", "a2"),
        "bcc_b2": (_b2(), "bcc", "b2"),
        "hcp": (bulk("Ti", "hcp", a=2.95, c=4.68, orthorhombic=True).repeat((3, 3, 3)), "hcp", "pure"),
        "sc": (bulk("Po", "sc", a=3.35, cubic=True).repeat((4, 4, 4)), "sc", "pure"),
        "diamond": (bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2)), "diamond", "pure"),
    }


def _nearest_distance(atoms: Atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    positive = distances[distances > 1.0e-8]
    return float(np.min(positive))


def _distort(
    atoms: Atoms,
    seed: int,
    *,
    strain: float,
    noise: float,
    vacancies: float = 0.0,
    shear: float = 0.0,
) -> Atoms:
    rng = np.random.default_rng(seed)
    result = atoms.copy()
    random_matrix = rng.normal(size=(3, 3))
    symmetric = 0.5 * (random_matrix + random_matrix.T)
    symmetric /= max(float(np.linalg.norm(symmetric, ord=2)), 1.0)
    deformation = np.eye(3) + strain * symmetric
    if shear:
        upper = np.triu(rng.uniform(-shear, shear, size=(3, 3)), k=1)
        deformation += upper
    if np.linalg.det(deformation) <= 0.2:
        raise RuntimeError("invalid synthetic deformation")
    result.set_cell(np.asarray(result.cell) @ deformation.T, scale_atoms=True)
    nearest = _nearest_distance(result)
    if noise:
        result.positions += rng.normal(scale=noise * nearest, size=result.positions.shape)
    if vacancies:
        remove_count = min(len(result) - 1, int(round(vacancies * len(result))))
        if remove_count:
            del result[np.sort(rng.choice(len(result), remove_count, replace=False))]
    result.wrap()
    return result


def build_training_frames(seed: int = 7) -> list[Frame]:
    conditions = (
        (0.000, 0.000, 0.00, 0.00),
        (0.025, 0.020, 0.00, 0.02),
        (0.050, 0.045, 0.00, 0.04),
        (0.075, 0.065, 0.00, 0.08),
        (0.110, 0.090, 0.00, 0.12),
        (0.055, 0.050, 0.04, 0.04),
        (0.080, 0.050, 0.02, 0.16),
        (0.115, 0.080, 0.03, 0.14),
    )
    frames: list[Frame] = []
    for source_index, (source, (base, geometry, ordering)) in enumerate(_base_structures(seed).items()):
        for replica, (strain, noise, vacancies, shear) in enumerate(conditions):
            if source == "fcc_a1":
                base = _fcc_a1(
                    seed + 997 * replica,
                    species_count=2 + replica % 3,
                    varied_composition=replica != 0,
                )
            elif source == "bcc_a2":
                base = _bcc_a2(
                    seed + 997 * replica,
                    species_count=2 + replica % 3,
                    varied_composition=replica != 0,
                )
            frames.append(
                Frame(
                    atoms=_distort(
                        base,
                        seed * 10000 + source_index * 100 + replica,
                        strain=strain,
                        noise=noise,
                        vacancies=vacancies,
                        shear=shear,
                    ),
                    geometry=geometry,
                    ordering=ordering,
                    condition="train",
                    source=source,
                )
            )
    return frames


def build_test_frames(seed: int = 91) -> list[Frame]:
    conditions = (
        ("clean", 0.015, 0.008, 0.0, 0.0),
        ("thermal", 0.045, 0.060, 0.0, 0.03),
        ("strong", 0.100, 0.095, 0.0, 0.08),
        ("vacancy", 0.060, 0.055, 0.05, 0.04),
        ("triclinic", 0.075, 0.045, 0.02, 0.16),
    )
    frames: list[Frame] = []
    for source_index, (source, (base, geometry, ordering)) in enumerate(_base_structures(seed).items()):
        for condition_index, (condition, strain, noise, vacancies, shear) in enumerate(conditions):
            frames.append(
                Frame(
                    atoms=_distort(
                        base,
                        seed * 10000 + source_index * 100 + condition_index,
                        strain=strain,
                        noise=noise,
                        vacancies=vacancies,
                        shear=shear,
                    ),
                    geometry=geometry,
                    ordering=ordering,
                    condition=condition,
                    source=source,
                )
            )
    for replica in range(5):
        a15 = _distort(_a15(), seed * 20000 + replica, strain=0.06, noise=0.05, shear=0.10)
        frames.append(Frame(a15, "unknown", "unknown", "unknown-a15", "a15"))
        amorphous = _amorphous(seed * 30000 + replica)
        frames.append(Frame(amorphous, "unknown", "unknown", "unknown-amorphous", "amorphous"))
        l11 = _distort(
            _l11(),
            seed * 40000 + replica,
            strain=0.04 + 0.01 * replica,
            noise=0.025 + 0.006 * replica,
            shear=0.03 + 0.01 * replica,
        )
        frames.append(Frame(l11, "fcc", "unknown", "unknown-order-l11", "fcc_l11"))
    return frames


def _sketch_frame(frame: Frame):
    atoms = frame.atoms
    return phase_sketch(
        np.asarray(atoms.positions),
        np.asarray(atoms.cell),
        np.asarray(atoms.pbc),
        np.asarray(atoms.numbers),
    )


def _dominant(labels: Iterable[str], *, minimum_fraction: float = 0.5) -> str:
    values = tuple(str(label) for label in labels)
    if not values:
        return "unknown"
    counts = Counter(values)
    label, count = counts.most_common(1)[0]
    return label if count / len(values) >= minimum_fraction else "unknown"


@dataclass
class PhaseSketchModel:
    structure_geometry: PrototypeBank
    structure_fcc_ordering: PrototypeBank
    structure_bcc_ordering: PrototypeBank
    geometry: PrototypeBank
    fcc_ordering: PrototypeBank
    bcc_ordering: PrototypeBank
    geometry_forest: ExtraTreesClassifier
    fcc_ordering_forest: ExtraTreesClassifier
    bcc_ordering_forest: ExtraTreesClassifier

    def predict(self, sketch) -> tuple[str, str]:
        return self.predict_many((sketch,))[0]

    def predict_many(self, sketches) -> list[tuple[str, str]]:
        """Classify a frame batch while keeping open-set decisions per frame."""
        predictions, _ = self.predict_many_with_evidence(sketches)
        return predictions

    def predict_many_with_evidence(
        self,
        sketches,
    ) -> tuple[list[tuple[str, str]], list[GeometryEvidence]]:
        """Classify a batch and retain why each geometry label was accepted."""
        sketches = tuple(sketches)
        if not sketches:
            return [], []
        geometry_summaries = np.stack(
            [summarize_phase_sketch(sketch.geometry) for sketch in sketches]
        )
        candidates = self.geometry_forest.predict(geometry_summaries).astype(object)
        structure_classes, structure_distances = (
            self.structure_geometry.distances_by_class(geometry_summaries)
        )
        structure_class_indices = {
            label: index for index, label in enumerate(structure_classes)
        }
        structure_ratios = np.asarray(
            [
                structure_distances[row, structure_class_indices[str(candidate)]]
                / self.structure_geometry.thresholds_[str(candidate)]
                for row, candidate in enumerate(candidates)
            ],
            dtype=np.float64,
        )
        structure_accepted = structure_ratios <= 1.0
        forest_probabilities = self.geometry_forest.predict_proba(geometry_summaries)
        forest_class_indices = {
            str(label): index
            for index, label in enumerate(self.geometry_forest.classes_)
        }
        candidate_probabilities = np.asarray(
            [
                forest_probabilities[row, forest_class_indices[str(candidate)]]
                for row, candidate in enumerate(candidates)
            ],
            dtype=np.float64,
        )
        diffuse_structure = np.asarray(
            [
                sketch.translational_order_score is not None
                and sketch.translational_order_limit is not None
                and sketch.translational_order_limit < 1.0
                and sketch.translational_order_score
                < sketch.translational_order_limit
                for sketch in sketches
            ],
            dtype=bool,
        )
        local_rows = np.flatnonzero(
            structure_accepted
            & ~diffuse_structure
            & (candidate_probabilities < _LOCAL_EVIDENCE_FOREST_PROBABILITY)
        )
        local_predictions: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if len(local_rows):
            local_lengths = np.asarray(
                [len(sketches[row].geometry) for row in local_rows], dtype=np.intp
            )
            local_prediction = self.geometry.predict(
                np.concatenate(
                    [sketches[row].geometry for row in local_rows], axis=0
                )
            )
            cursor = 0
            for row, length in zip(local_rows, local_lengths):
                stop = cursor + int(length)
                local_predictions[int(row)] = (
                    local_prediction.labels[cursor:stop],
                    local_prediction.margins[cursor:stop],
                    local_prediction.distances[cursor:stop],
                )
                cursor = stop

        geometries = candidates.copy()
        evidence: list[GeometryEvidence] = []
        for row, (candidate, accepted, structure_ratio, forest_probability) in enumerate(
            zip(
                candidates,
                structure_accepted,
                structure_ratios,
                candidate_probabilities,
            )
        ):
            cna_codes = sketches[row].cna_labels
            cna_fractions = {
                "fcc": float(np.mean(cna_codes == 1)),
                "hcp": float(np.mean(cna_codes == 2)),
                "bcc": float(np.mean(cna_codes == 3)),
                "other_or_unresolved": float(np.mean(cna_codes == 0)),
            }
            local_values = local_predictions.get(row)
            if local_values is None:
                local_support = float("nan")
                local_unknown = float("nan")
                local_margin = float("nan")
                local_distance = float("nan")
            else:
                local_labels, local_margins, local_distances = local_values
                local_support = float(np.mean(local_labels == candidate))
                local_unknown = float(np.mean(local_labels == "unknown"))
                local_margin = float(np.median(local_margins))
                local_distance = float(np.median(local_distances))
            if diffuse_structure[row]:
                label = "unknown"
                confidence_state = "diffuse_structure"
            elif not accepted:
                cna_support = cna_fractions.get(
                    str(candidate), 0.0
                )
                prototype_support = (
                    local_support if local_values is not None else 0.0
                )
                if max(prototype_support, cna_support) >= 0.80:
                    label = str(candidate)
                    confidence_state = "matched_local"
                else:
                    label = "unknown"
                    confidence_state = "outside_reference"
            elif (
                local_values is not None
                and local_support < _MINIMUM_LOCAL_GEOMETRY_SUPPORT
            ):
                label = "unknown"
                confidence_state = "low_local_support"
            elif (
                str(candidate) in {"fcc", "hcp"}
                and cna_fractions["other_or_unresolved"] <= 0.50
                and cna_fractions["fcc"] >= 0.20
                and cna_fractions["hcp"] >= 0.20
            ):
                label = str(candidate)
                confidence_state = "mixed_local"
            else:
                label = str(candidate)
                confidence_state = "matched"
            geometries[row] = label
            evidence.append(
                GeometryEvidence(
                    label=label,
                    candidate=str(candidate),
                    confidence_state=confidence_state,
                    local_support_fraction=local_support,
                    local_unknown_fraction=local_unknown,
                    local_margin_median=local_margin,
                    local_distance_median=local_distance,
                    local_evidence_evaluated=local_values is not None,
                    structure_distance_ratio=float(structure_ratio),
                    forest_probability=float(forest_probability),
                    translational_order_score=(
                        float(sketches[row].translational_order_score)
                        if sketches[row].translational_order_score is not None
                        else None
                    ),
                    translational_order_limit=(
                        float(sketches[row].translational_order_limit)
                        if sketches[row].translational_order_limit is not None
                        else None
                    ),
                    cna_phase_fractions=cna_fractions,
                )
            )

        orderings = np.full(len(sketches), "pure", dtype=object)
        for geometry, forest, bank in (
            ("fcc", self.fcc_ordering_forest, self.structure_fcc_ordering),
            ("bcc", self.bcc_ordering_forest, self.structure_bcc_ordering),
        ):
            rows = np.flatnonzero(geometries == geometry)
            if not len(rows):
                continue
            chemistry_summaries = np.stack(
                [summarize_phase_sketch(sketches[row].chemistry) for row in rows]
            )
            labels = forest.predict(chemistry_summaries).astype(object)
            accepted = bank.accepts_labels(chemistry_summaries, labels)
            labels[~accepted] = "unknown"
            orderings[rows] = labels
        orderings[geometries == "unknown"] = "unknown"
        predictions = [
            (str(geometry), str(ordering))
            for geometry, ordering in zip(geometries, orderings)
        ]
        return predictions, evidence

def fit_phase_sketch(frames: list[Frame]) -> tuple[PhaseSketchModel, list]:
    sketches = [_sketch_frame(frame) for frame in frames]
    structure_geometry = PrototypeBank(
        prototypes_per_class=4,
        samples_per_prototype=2,
        rejection_scale=4.2,
        minimum_margin=1.0,
    )
    structure_geometry.fit(
        np.stack([summarize_phase_sketch(value.geometry) for value in sketches]),
        [frame.geometry for frame in frames],
    )
    geometry_summaries = np.stack([summarize_phase_sketch(value.geometry) for value in sketches])
    geometry_forest = ExtraTreesClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=17,
        n_jobs=1,
    ).fit(geometry_summaries, [frame.geometry for frame in frames])
    geometry_values = np.concatenate([value.geometry for value in sketches])
    geometry_labels = [label for frame, value in zip(frames, sketches) for label in [frame.geometry] * len(value.geometry)]
    geometry = PrototypeBank(prototypes_per_class=16, rejection_scale=1.55, minimum_margin=1.005)
    geometry.fit(geometry_values, geometry_labels)

    fcc_values = np.concatenate([value.chemistry for frame, value in zip(frames, sketches) if frame.geometry == "fcc"])
    fcc_labels = [label for frame, value in zip(frames, sketches) if frame.geometry == "fcc" for label in [frame.ordering] * len(value.chemistry)]
    fcc_ordering = PrototypeBank(prototypes_per_class=16, rejection_scale=1.7, minimum_margin=1.002)
    fcc_ordering.fit(fcc_values, fcc_labels)

    structure_fcc_ordering = PrototypeBank(
        prototypes_per_class=4,
        samples_per_prototype=2,
        rejection_scale=6.0,
        minimum_margin=1.0,
    )
    structure_fcc_ordering.fit(
        np.stack(
            [summarize_phase_sketch(value.chemistry) for frame, value in zip(frames, sketches) if frame.geometry == "fcc"]
        ),
        [frame.ordering for frame in frames if frame.geometry == "fcc"],
    )
    fcc_structure_values = np.stack(
        [summarize_phase_sketch(value.chemistry) for frame, value in zip(frames, sketches) if frame.geometry == "fcc"]
    )
    fcc_structure_labels = [frame.ordering for frame in frames if frame.geometry == "fcc"]
    fcc_ordering_forest = ExtraTreesClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=23,
        n_jobs=1,
    ).fit(fcc_structure_values, fcc_structure_labels)

    bcc_values = np.concatenate([value.chemistry for frame, value in zip(frames, sketches) if frame.geometry == "bcc"])
    bcc_labels = [label for frame, value in zip(frames, sketches) if frame.geometry == "bcc" for label in [frame.ordering] * len(value.chemistry)]
    bcc_ordering = PrototypeBank(prototypes_per_class=12, rejection_scale=1.7, minimum_margin=1.002)
    bcc_ordering.fit(bcc_values, bcc_labels)
    structure_bcc_ordering = PrototypeBank(
        prototypes_per_class=4,
        samples_per_prototype=2,
        rejection_scale=4.2,
        minimum_margin=1.0,
    )
    structure_bcc_ordering.fit(
        np.stack(
            [summarize_phase_sketch(value.chemistry) for frame, value in zip(frames, sketches) if frame.geometry == "bcc"]
        ),
        [frame.ordering for frame in frames if frame.geometry == "bcc"],
    )
    bcc_structure_values = np.stack(
        [summarize_phase_sketch(value.chemistry) for frame, value in zip(frames, sketches) if frame.geometry == "bcc"]
    )
    bcc_structure_labels = [frame.ordering for frame in frames if frame.geometry == "bcc"]
    bcc_ordering_forest = ExtraTreesClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=29,
        n_jobs=1,
    ).fit(bcc_structure_values, bcc_structure_labels)
    return PhaseSketchModel(
        structure_geometry,
        structure_fcc_ordering,
        structure_bcc_ordering,
        geometry,
        fcc_ordering,
        bcc_ordering,
        geometry_forest,
        fcc_ordering_forest,
        bcc_ordering_forest,
    ), sketches


def _ptm_prediction(
    frame: Frame,
    *,
    rmsd_cutoff: float = 0.10,
) -> tuple[str, str]:
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    from ovito.pipeline import Pipeline, StaticSource

    modifier = PolyhedralTemplateMatchingModifier(
        output_rmsd=True,
        output_ordering=True,
        rmsd_cutoff=rmsd_cutoff,
    )
    for structure_type in modifier.structures:
        structure_type.enabled = True
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(frame.atoms)))
    pipeline.modifiers.append(modifier)
    output = pipeline.compute()
    structure_types = np.asarray(output.particles["Structure Type"], dtype=np.int32)
    mapping = {0: "unknown", 1: "fcc", 2: "hcp", 3: "bcc", 4: "ico", 5: "sc", 6: "diamond", 7: "diamond", 8: "graphene"}
    geometry = _dominant(mapping.get(int(value), "unknown") for value in structure_types)
    if geometry not in {"fcc", "bcc"}:
        return geometry, "unknown" if geometry == "unknown" else "pure"
    ordering_types = np.asarray(output.particles["Ordering Type"], dtype=np.int32)
    if geometry == "fcc":
        ordering_mapping = {0: "a1", 1: "pure", 2: "l10", 3: "l12", 4: "l12"}
    else:
        ordering_mapping = {0: "a2", 1: "pure", 5: "b2"}
    ordering = _dominant(ordering_mapping.get(int(value), "unknown") for value in ordering_types)
    return geometry, ordering


def _phase_label(geometry: str, ordering: str) -> str:
    if geometry in {"fcc", "bcc"}:
        return f"{geometry}:{ordering}"
    return geometry


def _scores(truth: list[str], predicted: list[str]) -> dict[str, object]:
    labels = sorted(set(truth) | set(predicted))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def run_benchmark(repeats: int = 3) -> dict[str, object]:
    training = build_training_frames()
    test = build_test_frames()
    train_start = time.perf_counter()
    model, _ = fit_phase_sketch(training)
    training_seconds = time.perf_counter() - train_start

    sketch_cache = [_sketch_frame(frame) for frame in test]
    phase_sketch_predictions, geometry_evidence = model.predict_many_with_evidence(
        sketch_cache
    )
    ptm_predictions = [_ptm_prediction(frame) for frame in test]

    truth_geometry = [frame.geometry for frame in test]
    truth_phase = [frame.phase for frame in test]
    sketch_geometry = [value[0] for value in phase_sketch_predictions]
    sketch_phase = [_phase_label(*value) for value in phase_sketch_predictions]
    ptm_geometry = [value[0] for value in ptm_predictions]
    ptm_phase = [_phase_label(*value) for value in ptm_predictions]

    atom_count = sum(len(frame.atoms) for frame in test)
    sketch_times: list[float] = []
    ptm_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_many([_sketch_frame(frame) for frame in test])
        sketch_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        for frame in test:
            _ptm_prediction(frame)
        ptm_times.append(time.perf_counter() - started)

    by_condition: dict[str, dict[str, float]] = {}
    for condition in sorted({frame.condition for frame in test}):
        rows = [index for index, frame in enumerate(test) if frame.condition == condition]
        by_condition[condition] = {
            "phase_sketch_accuracy": float(accuracy_score([truth_phase[i] for i in rows], [sketch_phase[i] for i in rows])),
            "ptm_accuracy": float(accuracy_score([truth_phase[i] for i in rows], [ptm_phase[i] for i in rows])),
        }

    unknown_rows = [index for index, frame in enumerate(test) if frame.geometry == "unknown"]
    unknown_order_rows = [
        index
        for index, frame in enumerate(test)
        if frame.geometry in {"fcc", "bcc"} and frame.ordering == "unknown"
    ]
    return {
        "dataset": {
            "training_frames": len(training),
            "test_frames": len(test),
            "test_atoms": atom_count,
            "known_test_frames": len(test) - len(unknown_rows),
            "unknown_test_frames": len(unknown_rows),
        },
        "phase_sketch": {
            "backend": (
                "cpp-openmp"
                if phase_sketch_module._native_phase is not None
                else "python-reference"
            ),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "geometry": _scores(truth_geometry, sketch_geometry),
            "ordered_phase": _scores(truth_phase, sketch_phase),
            "unknown_recall": float(np.mean([sketch_geometry[index] == "unknown" for index in unknown_rows])),
            "unknown_order_recall": float(
                np.mean([sketch_phase[index].endswith(":unknown") for index in unknown_order_rows])
            ),
            "confidence_state": dict(
                sorted(Counter(value.confidence_state for value in geometry_evidence).items())
            ),
            "training_seconds": training_seconds,
            "median_inference_seconds": median(sketch_times),
            "atoms_per_second": atom_count / median(sketch_times),
        },
        "ptm": {
            "geometry": _scores(truth_geometry, ptm_geometry),
            "ordered_phase": _scores(truth_phase, ptm_phase),
            "unknown_recall": float(np.mean([ptm_geometry[index] == "unknown" for index in unknown_rows])),
            "unknown_order_recall": float(
                np.mean([ptm_phase[index].endswith(":unknown") for index in unknown_order_rows])
            ),
            "median_inference_seconds": median(ptm_times),
            "atoms_per_second": atom_count / median(ptm_times),
        },
        "by_condition": by_condition,
        "predictions": [
            {
                "source": frame.source,
                "condition": frame.condition,
                "truth": truth_phase[index],
                "phase_sketch": sketch_phase[index],
                "phase_candidate": geometry_evidence[index].candidate,
                "confidence_state": geometry_evidence[index].confidence_state,
                "local_support_fraction": (
                    geometry_evidence[index].local_support_fraction
                    if geometry_evidence[index].local_evidence_evaluated
                    else None
                ),
                "cna_phase_fractions": geometry_evidence[
                    index
                ].cna_phase_fractions,
                "ptm": ptm_phase[index],
            }
            for index, frame in enumerate(test)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="disable the optional C++/OpenMP backend",
    )
    args = parser.parse_args()
    if args.reference:
        phase_sketch_module._native_phase = None
    result = run_benchmark(max(1, args.repeats))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
