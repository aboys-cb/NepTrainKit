import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from tools.benchmark_phase_sketch import (
    Frame,
    _sketch_frame,
    build_training_frames,
    fit_phase_sketch,
)


def _stacked_close_packed(sequence: str, size: int = 4) -> Atoms:
    distance = 2.55
    first = np.asarray((distance, 0.0, 0.0))
    second = np.asarray((0.5 * distance, np.sqrt(3.0) * 0.5 * distance, 0.0))
    layer_height = np.sqrt(2.0 / 3.0) * distance
    offsets = {
        "A": np.zeros(3),
        "B": (first + second) / 3.0,
        "C": 2.0 * (first + second) / 3.0,
    }
    positions = [
        row * first
        + column * second
        + offsets[layer]
        + (0.0, 0.0, depth * layer_height)
        for depth, layer in enumerate(sequence)
        for row in range(size)
        for column in range(size)
    ]
    return Atoms(
        f"Cu{len(positions)}",
        positions=positions,
        cell=(
            size * first,
            size * second,
            (0.0, 0.0, len(sequence) * layer_height),
        ),
        pbc=True,
    )


def test_phase_model_exposes_hcp_interface_and_stacking_fault_fractions():
    model, _ = fit_phase_sketch(build_training_frames())
    structures = (
        _stacked_close_packed("AB" * 12),
        _stacked_close_packed("ABC" * 4 + "AB" * 6),
        _stacked_close_packed("ABCABCABABCABCABCABCABC"),
        bulk("Fe", "bcc", a=2.87, cubic=True).repeat((4, 4, 4)),
    )
    sketches = [
        _sketch_frame(Frame(atoms, "unknown", "unknown", "test", str(index)))
        for index, atoms in enumerate(structures)
    ]

    predictions, evidence = model.predict_many_with_evidence(sketches)

    assert predictions == [
        ("hcp", "pure"),
        ("fcc", "pure"),
        ("fcc", "pure"),
        ("bcc", "pure"),
    ]
    assert [value.confidence_state for value in evidence] == [
        "matched_local",
        "mixed_local",
        "matched",
        "matched",
    ]
    assert evidence[0].cna_phase_fractions["hcp"] == 1.0
    assert evidence[1].cna_phase_fractions["fcc"] == 0.5
    assert evidence[1].cna_phase_fractions["hcp"] == 0.5
    assert evidence[2].cna_phase_fractions["hcp"] == pytest.approx(
        2.0 / 23.0
    )
    assert evidence[3].cna_phase_fractions["bcc"] == 1.0
