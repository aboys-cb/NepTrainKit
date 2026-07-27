from __future__ import annotations

import sys
import threading

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


def _escaped_json_trajectory() -> bytes:
    return (
        '1\n'
        'Lattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3 '
        'rss_composition="_JSON {\\"B\\": 0.6, \\"N\\": 0.4}" '
        'prerelax_energy=-4.5 stress="1 0 0 0 1 0 0 0 1" '
        'energy=-10.25 pbc="T T T"\n'
        'B 0 0 0\n'
    ).encode("ascii")


def _single_atom_trajectory(frame_count: int) -> bytes:
    frame = (
        b"1\n"
        b'Properties=species:S:1:pos:R:3 Lattice="3 0 0 0 3 0 0 0 3"\n'
        b"Fe 0 0 0\n"
    )
    return frame * frame_count


def test_parallel_parse_keeps_pos_first_frame_boundaries(monkeypatch):
    fastxyz = pytest.importorskip("NepTrainKit._native._io")
    monkeypatch.setenv("NEPKIT_FASTXYZ_SPECIES_MODE", "str")

    frames = fastxyz.parse_all(memoryview(_pos_first_trajectory()), 4)

    assert len(frames) == 61
    assert all(frame["lattice"].shape == (9,) for frame in frames)
    assert frames[0]["atomic_properties"]["pos"].shape == (47, 3)
    assert frames[0]["atomic_properties"]["species"][0] == "Fe"
    assert all(
        frame["additional_fields"]["pbc"] == "T T T" for frame in frames
    )


def test_parse_escaped_json_keeps_following_fields(monkeypatch):
    fastxyz = pytest.importorskip("NepTrainKit._native._io")
    monkeypatch.setenv("NEPKIT_FASTXYZ_SPECIES_MODE", "str")

    frame = fastxyz.parse_all(memoryview(_escaped_json_trajectory()), 1)[0]
    additional_fields = frame["additional_fields"]

    assert additional_fields["rss_composition"] == '_JSON {"B": 0.6, "N": 0.4}'
    assert additional_fields["prerelax_energy"] == "-4.5"
    assert additional_fields["energy"] == -10.25
    assert additional_fields["stress"].shape == (9,)
    assert additional_fields["pbc"] == "T T T"


def test_index_frames_yields_the_gil_during_large_scans():
    fastxyz = pytest.importorskip("NepTrainKit._native._io")
    payload = memoryview(_single_atom_trajectory(250_000))
    observer_ready = threading.Event()
    start = threading.Event()
    observed = threading.Event()

    def observe_released_gil():
        observer_ready.set()
        start.wait()
        observed.set()

    observer = threading.Thread(target=observe_released_gil)
    observer.start()
    assert observer_ready.wait(timeout=5)

    previous_switch_interval = sys.getswitchinterval()
    try:
        # Prevent an ordinary Python bytecode timeslice from satisfying the
        # assertion.  The observer can run here only while index_frames has
        # explicitly released the GIL around its native scan.
        sys.setswitchinterval(60.0)
        start.set()
        frames = fastxyz.index_frames(payload)
        observed_during_call = observed.is_set()
    finally:
        sys.setswitchinterval(previous_switch_interval)
        start.set()
        observer.join(timeout=5)

    assert not observer.is_alive()
    assert len(frames) == 250_000
    assert observed_during_call
