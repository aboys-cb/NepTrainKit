from __future__ import annotations

import pytest


def _pos_first_trajectory(frame_count: int = 61) -> bytes:
    lines: list[str] = []
    for frame in range(frame_count):
        atom_count = 47 + frame % 7
        lines.append(str(atom_count))
        lines.append(
            'Lattice="10 0 0 0 11 0 0 0 12" '
            f'frame={frame} Properties=pos:R:3:species:S:1'
        )
        for atom in range(atom_count):
            lines.append(f"1.{atom:02d} 2.{atom:02d} 3.{atom:02d} Fe")
    return ("\n".join(lines) + "\n").encode("ascii")


def test_parallel_parse_keeps_pos_first_frame_boundaries(monkeypatch):
    fastxyz = pytest.importorskip("NepTrainKit.core._fastxyz")
    monkeypatch.setenv("NEPKIT_FASTXYZ_SPECIES_MODE", "str")

    frames = fastxyz.parse_all(memoryview(_pos_first_trajectory()), 4)

    assert len(frames) == 61
    assert all(frame["lattice"].shape == (9,) for frame in frames)
    assert frames[0]["atomic_properties"]["pos"].shape == (47, 3)
    assert frames[0]["atomic_properties"]["species"][0] == "Fe"
