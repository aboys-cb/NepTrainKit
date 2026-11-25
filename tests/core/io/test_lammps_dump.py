"""LAMMPS dump probing, column plans, tolerant frames and element resolution."""
from __future__ import annotations

import gzip

import numpy as np
import pytest

from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.io.lammps_dump import (
    LammpsDumpError,
    iter_dump_frames,
    load_dump_structures,
    probe_dump,
    suggest_element_mapping,
)


def _frame(
    columns: str,
    rows: list[str],
    *,
    timestep: int = 0,
    n_atoms: int | None = None,
    bounds: tuple[tuple[str, ...], ...] = (("0", "10"), ("0", "10"), ("0", "10")),
    bounds_header: str = "pp pp pp",
) -> str:
    lines = [
        "ITEM: TIMESTEP",
        str(timestep),
        "ITEM: NUMBER OF ATOMS",
        str(len(rows) if n_atoms is None else n_atoms),
        f"ITEM: BOX BOUNDS {bounds_header}",
    ]
    lines.extend(" ".join(row) for row in bounds)
    lines.append(f"ITEM: ATOMS {columns}")
    lines.extend(rows)
    return "\n".join(lines) + "\n"


def _write(path, text: str):
    path.write_text(text, encoding="utf8")
    return path


def test_probe_reports_frames_columns_and_types(tmp_path):
    path = _write(
        tmp_path / "traj.dump",
        _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"], timestep=0)
        + _frame("id type x y z", ["1 1 0.1 0 0", "2 2 1.1 1 1"], timestep=10),
    )

    probe = probe_dump(path)

    assert probe.frame_count == 2
    assert probe.declared_frame_count == 2
    assert probe.atom_counts == (2, 2)
    assert probe.timesteps == (0, 10)
    assert probe.types == (1, 2)
    assert probe.species_source == "type"
    assert probe.needs_element_mapping is True
    assert probe.columns == ("id", "type", "x", "y", "z")
    assert probe.cell_styles == ("ortho",)
    assert probe.truncated is False


def test_typelabel_column_is_accepted_as_species(tmp_path):
    path = _write(
        tmp_path / "labels.dump",
        _frame("id typelabel x y z", ["1 Fe 0 0 0", "2 Ni 1 1 1"]),
    )

    structures = import_structures(path)

    assert probe_dump(path).species_source == "typelabel"
    assert structures[0].elements.tolist() == ["Fe", "Ni"]


def test_extra_columns_are_preserved(tmp_path):
    path = _write(
        tmp_path / "extras.dump",
        _frame(
            "id element x y z vx vy vz q c_pe",
            ["1 Fe 0 0 0 0.1 0.2 0.3 -1.0 -5.1", "2 Fe 1 1 1 0 0 0 1.0 -5.2"],
        ),
    )

    structure = import_structures(path)[0]
    names = {prop["name"] for prop in structure.properties}

    assert {"vel", "dump.q", "dump.c_pe"} <= names
    np.testing.assert_allclose(structure.atomic_properties["vel"], [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(structure.atomic_properties["dump.q"], [-1.0, 1.0])

    without_extras = import_structures(path, keep_extra=False)[0]
    assert {prop["name"] for prop in without_extras.properties} == {"species", "pos"}


def test_three_column_spin_is_read_as_a_vector(tmp_path):
    path = _write(
        tmp_path / "spin3.dump",
        _frame("id element x y z c_sp[1] c_sp[2] c_sp[3]", ["1 Fe 0 0 0 0 0 1", "2 Fe 1 1 1 1 0 0"]),
    )

    structure = import_structures(path)[0]

    np.testing.assert_allclose(structure.atomic_properties["spin"], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def test_spin_compute_name_is_not_hard_coded(tmp_path):
    path = _write(
        tmp_path / "spin4.dump",
        _frame(
            "id element x y z c_sp[1] c_sp[2] c_sp[3] c_sp[4]",
            ["1 Fe 0 0 0 2.0 0.0 0.6 0.8", "2 Fe 1 1 1 1.5 -1.0 0.0 0.0"],
        ),
    )

    structure = import_structures(path)[0]

    np.testing.assert_allclose(
        structure.atomic_properties["spin"], [[0.0, 1.2, 1.6], [-1.5, 0.0, 0.0]], atol=1e-6
    )


def test_truncated_trailing_frame_is_dropped_with_a_warning(tmp_path):
    path = _write(
        tmp_path / "cut.dump",
        _frame("id element x y z", ["1 H 0 0 0", "2 He 1 1 1"], timestep=0)
        + _frame("id element x y z", ["1 H 0 0 0"], timestep=10, n_atoms=2),
    )
    warnings: list[str] = []

    assert probe_dump(path).frame_count == 1
    assert probe_dump(path).truncated is True
    frames = list(iter_dump_frames(path, on_warning=warnings.append))

    assert [frame.timestep for frame in frames] == [0]
    assert warnings and "truncated" in warnings[0]
    with pytest.raises(LammpsDumpError, match="expected 2 atom rows, found 1"):
        list(iter_dump_frames(path, strict=True))


def test_single_truncated_frame_still_reports_the_frame_error(tmp_path):
    path = _write(tmp_path / "cut_one.dump", _frame("id element x y z", [], n_atoms=1))

    with pytest.raises(LammpsDumpError, match="expected 1 atom rows, found 0"):
        load_dump_structures(path)


def test_data_file_masses_resolve_numeric_types(tmp_path):
    _write(
        tmp_path / "data.lammps",
        "LAMMPS data file\n\n2 atoms\n2 atom types\n\nMasses\n\n1 55.845\n2 28.0855\n",
    )
    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))

    suggestion = suggest_element_mapping(path)
    structures = import_structures(path)

    assert suggestion.missing == ()
    assert dict(suggestion.mapping) == {1: "Fe", 2: "Si"}
    assert structures[0].elements.tolist() == ["Fe", "Si"]


def test_data_file_atom_type_labels_resolve_numeric_types(tmp_path):
    _write(
        tmp_path / "data.lammps",
        "LAMMPS data file\n\n2 atoms\n2 atom types\n\nAtom Type Labels\n\n1 Fe\n2 Si\n",
    )
    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))

    assert import_structures(path)[0].elements.tolist() == ["Fe", "Si"]


def test_input_script_labelmap_resolves_numeric_types(tmp_path):
    _write(
        tmp_path / "in.mylammps",
        "units metal\nlabelmap atom 1 Fe 2 Si\ndump d all custom 100 typed.dump id type x y z\nrun 0\n",
    )
    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))

    assert import_structures(path)[0].elements.tolist() == ["Fe", "Si"]


def test_conflicting_data_files_do_not_guess(tmp_path):
    _write(tmp_path / "one.data", "LAMMPS data file\n\n2 atoms\n2 atom types\n\nAtom Type Labels\n\n1 Fe\n2 Si\n")
    _write(tmp_path / "two.data", "LAMMPS data file\n\n2 atoms\n2 atom types\n\nAtom Type Labels\n\n1 Ni\n2 C\n")
    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))

    with pytest.raises(ValueError, match=r"unresolved types: \[1, 2\]"):
        import_structures(path)


def test_non_standard_mass_is_not_guessed(tmp_path):
    _write(tmp_path / "coarse.data", "LAMMPS data file\n\n1 atoms\n1 atom types\n\nMasses\n\n1 1.0\n")
    path = _write(tmp_path / "coarse.dump", _frame("id type x y z", ["1 1 0 0 0"]))

    with pytest.raises(ValueError, match=r"unresolved types: \[1\]"):
        import_structures(path)


def test_gzip_dump_is_supported(tmp_path):
    payload = _frame("id element x y z", ["1 Si 1 1 1"])
    path = tmp_path / "traj.dump.gz"
    path.write_bytes(gzip.compress(payload.encode("utf8")))

    probe = probe_dump(path)
    structures = import_structures(path)

    assert probe.compressed == "gz"
    assert structures[0].elements.tolist() == ["Si"]


def test_general_triclinic_box_is_parsed(tmp_path):
    path = _write(
        tmp_path / "general.dump",
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "3",
                "ITEM: NUMBER OF ATOMS",
                "1",
                "ITEM: BOX BOUNDS abc origin",
                "4 0 0 1",
                "0 5 0 2",
                "0 0 6 3",
                "ITEM: ATOMS id element x y z",
                "1 Si 2 3 4",
                "",
            ]
        ),
    )

    structure = import_structures(path)[0]

    np.testing.assert_allclose(structure.lattice, np.diag([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(structure.positions, [[1.0, 1.0, 1.0]])


def test_image_flags_unwrap_positions_on_request(tmp_path):
    path = _write(
        tmp_path / "images.dump",
        _frame("id element x y z ix iy iz", ["1 Si 1 1 1 1 0 0"]),
    )

    wrapped = load_dump_structures(path)[0]
    unwrapped = load_dump_structures(path, unwrap=True)[0]

    np.testing.assert_allclose(wrapped.positions, [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(unwrapped.positions, [[11.0, 1.0, 1.0]])


def test_per_frame_column_drift_is_replanned(tmp_path):
    path = _write(
        tmp_path / "drift.dump",
        _frame("id element x y z", ["1 H 0 0 0"], timestep=0)
        + _frame("id element x y z fx fy fz", ["1 H 0 0 0 1 2 3"], timestep=5),
    )

    probe = probe_dump(path)
    structures = import_structures(path)

    assert probe.signatures == (("id", "element", "x", "y", "z"), ("id", "element", "x", "y", "z", "fx", "fy", "fz"))
    assert "forces" not in {prop["name"] for prop in structures[0].properties}
    np.testing.assert_allclose(structures[1].forces, [[1.0, 2.0, 3.0]])


def test_alternative_coordinate_flavours_are_not_kept_as_extras(tmp_path):
    path = _write(
        tmp_path / "both.dump",
        _frame("id element x y z xs ys zs", ["1 Fe 9 9 9 0.1 0.2 0.3"]),
    )

    structure = import_structures(path)[0]

    assert [prop["name"] for prop in structure.properties] == ["species", "pos"]
    np.testing.assert_allclose(structure.positions, [[1.0, 2.0, 3.0]])


def test_probe_tolerates_a_frame_header_cut_in_the_middle(tmp_path):
    path = _write(
        tmp_path / "cut_header.dump",
        _frame("id element x y z", ["1 H 0 0 0"], timestep=0) + "ITEM: TIMESTEP\n10\nITEM: NUM",
    )

    probe = probe_dump(path)

    assert probe.frame_count == 1
    assert probe.declared_frame_count == 2
    assert probe.truncated is True
    assert len(load_dump_structures(path)) == 1


def test_missing_coordinate_triplet_is_rejected(tmp_path):
    path = _write(tmp_path / "nocoords.dump", _frame("id element", ["1 H"]))

    with pytest.raises(ValueError, match="must contain one complete coordinate triplet"):
        import_structures(path)


def _load_with(monkeypatch, path, answer, *, saved=None):
    """Run ``OtherLoader.load`` with a stub result object and a scripted dialog."""
    from types import SimpleNamespace

    import NepTrainKit.core.io.lammps_dump as lammps
    from NepTrainKit.core.io import registry

    monkeypatch.setattr(lammps, "remembered_dump_settings", lambda *args, **kwargs: saved)
    monkeypatch.setattr(registry.NepTrainResultData, "from_path", staticmethod(lambda _path: SimpleNamespace()))
    calls: list[object] = []

    def fake_prompt(suggestion, overrides=None):
        calls.append(suggestion)
        return answer

    monkeypatch.setattr(registry.OtherLoader, "_prompt_element_mapping", staticmethod(fake_prompt))
    instance = registry.OtherLoader().load(path)
    return instance, calls


def test_loader_skips_the_prompt_when_the_dump_has_elements(tmp_path, monkeypatch):
    path = _write(tmp_path / "elements.dump", _frame("id element x y z", ["1 Fe 0 0 0"]))

    instance, calls = _load_with(monkeypatch, path, None)

    assert calls == []
    assert getattr(instance, "_import_options", {}) == {}


def test_loader_applies_discovered_mapping_without_a_prompt(tmp_path, monkeypatch):
    _write(tmp_path / "data.lammps", "LAMMPS data file\n\n1 atoms\n1 atom types\n\nMasses\n\n1 55.845\n")
    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0"]))

    instance, calls = _load_with(monkeypatch, path, None)

    assert calls == []
    assert instance._import_options["element_map"] == {1: "Fe"}


def test_loader_prefills_the_dialog_and_applies_its_options(tmp_path, monkeypatch):
    from types import SimpleNamespace

    _write(tmp_path / "partial.data", "LAMMPS data file\n\n2 atoms\n2 atom types\n\nAtom Type Labels\n\n1 Fe\n")
    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))
    answer = SimpleNamespace(
        element_map={2: "Si"}, overrides=None, unwrap=True, keep_extra=False, strict=True, remember=False
    )

    instance, calls = _load_with(monkeypatch, path, answer)

    assert len(calls) == 1
    assert calls[0].mapping == {1: "Fe"}
    assert calls[0].missing == (2,)
    assert instance._import_options == {
        "element_map": {1: "Fe", 2: "Si"},
        "unwrap": True,
        "keep_extra": False,
        "strict": True,
    }


def test_loader_prefills_from_the_last_mapping_and_still_prompts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import NepTrainKit.core.io.lammps_dump as module

    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))
    answer = SimpleNamespace(
        element_map={1: "Fe", 2: "Co"},
        overrides=None,
        unwrap=False,
        keep_extra=True,
        strict=False,
        remember=False,
    )

    instance, calls = _load_with(
        monkeypatch, path, answer, saved=module.DumpSettings({1: "Fe", 2: "Co"})
    )

    assert len(calls) == 1
    assert calls[0].mapping == {1: "Fe", 2: "Co"}
    assert calls[0].missing == ()
    assert instance._import_options["element_map"] == {1: "Fe", 2: "Co"}


def test_loader_remembers_the_confirmed_mapping(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import NepTrainKit.core.io.lammps_dump as module

    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))
    remembered: list[tuple] = []
    monkeypatch.setattr(module, "remembered_dump_settings", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "remember_dump_settings",
        lambda mapping, **kwargs: remembered.append((mapping, kwargs)),
    )
    answer = SimpleNamespace(
        element_map={1: "Fe", 2: "Co"},
        overrides=None,
        unwrap=False,
        keep_extra=True,
        strict=False,
        remember=True,
    )

    _load_with(monkeypatch, path, answer)

    assert remembered
    mapping, kwargs = remembered[0]
    assert mapping == {1: "Fe", 2: "Co"}
    assert kwargs["signature"] == ("id", "type", "x", "y", "z")


def test_loader_raises_a_dedicated_cancellation(tmp_path, monkeypatch):
    from NepTrainKit.core.io.lammps_dump import DumpImportCancelled

    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0"]))

    with pytest.raises(DumpImportCancelled):
        _load_with(monkeypatch, path, None)


def _frame_signature(frame):
    return (
        frame.timestep,
        np.asarray(frame.lattice).tolist(),
        frame.pbc,
        frame.species.tolist(),
        np.asarray(frame.positions).tolist(),
        None if frame.forces is None else np.asarray(frame.forces).tolist(),
        None if frame.spins is None else np.asarray(frame.spins).tolist(),
        {name: np.asarray(values).tolist() for name, values in (frame.extras or {}).items()},
    )


def _rich_trajectory(tmp_path):
    header = "ITEM: BOX BOUNDS xy xz yz pp pp ff"
    bounds = "0 7 2\n0 6 1\n0 4 -1"
    columns = "id element x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] vx vy vz q"
    rows = [
        "1 Fe 2.5 2.5 2.0 2.0 0.0 0.6 0.8 0.1 0.2 0.3 -1.0",
        "2 Ni 1.25 0.5 1.0 1.5 -1.0 0.0 0.0 0.0 0.0 0.0 1.0",
    ]
    prefix = ["ITEM: NUMBER OF ATOMS", "2", header, bounds, f"ITEM: ATOMS {columns}"]
    first = "\n".join(["ITEM: TIMESTEP", "3", *prefix, *rows])
    second = "\n".join(["ITEM: TIMESTEP", "8", *prefix, *rows])
    return _write(tmp_path / "rich.dump", first + "\n" + second + "\n")


def test_native_and_python_parsers_agree(tmp_path, monkeypatch):
    import NepTrainKit.core.io.lammps_dump as module

    if module._native_dump() is None:
        pytest.skip("native dump parser is not built")
    path = _rich_trajectory(tmp_path)

    native_frames = [_frame_signature(frame) for frame in iter_dump_frames(path)]
    monkeypatch.setattr(module, "_native_dump", lambda: None)
    python_frames = [_frame_signature(frame) for frame in iter_dump_frames(path)]

    assert native_frames == python_frames


@pytest.mark.parametrize(
    ("columns", "rows", "n_atoms", "message", "more_frames"),
    [
        ("id element x y z fx fy fz", ["1 Fe 0 0 0 1 2 3", "2 Fe 0 0 1 2"], 2, "has 6 columns; expected 8", True),
        ("id element x y z", ["1 Fe 0 0 0", "2 Fe 0 0 bad"], 2, "contains a non-numeric value", False),
        ("id element x y z", ["1 Fe 0 0 0", "2 Fe 0 0 0"], 1, "more atom rows than the declared 1", False),
    ],
)
def test_native_error_paths_match_the_python_fallback(
    tmp_path, monkeypatch, columns, rows, n_atoms, message, more_frames
):
    import NepTrainKit.core.io.lammps_dump as module

    body = _frame(columns, rows, n_atoms=n_atoms)
    if more_frames:
        body += _frame(columns, ["1 Fe 0 0 0 1 2 3"], n_atoms=1, timestep=10)
    path = _write(tmp_path / "broken.dump", body)

    with pytest.raises(LammpsDumpError, match=message):
        list(iter_dump_frames(path))
    monkeypatch.setattr(module, "_native_dump", lambda: None)
    with pytest.raises(LammpsDumpError, match=message):
        list(iter_dump_frames(path))


def test_parallel_batches_match_sequential_parsing(tmp_path, monkeypatch):
    import NepTrainKit.core.io.lammps_dump as module

    if module._native_dump() is None:
        pytest.skip("native dump parser is not built")
    rows = [f"{index} Si 0.1 0.2 0.3 1 2 3" for index in range(7000)]
    body = "".join(_frame("id element x y z fx fy fz", rows, timestep=step) for step in range(4))
    path = _write(tmp_path / "parallel.dump", body)

    sequential = [_frame_signature(frame) for frame in iter_dump_frames(path, max_workers=1)]
    parallel = [_frame_signature(frame) for frame in iter_dump_frames(path, max_workers=4)]

    assert sequential == parallel
    monkeypatch.setattr(module, "_native_dump", lambda: None)
    fallback = [_frame_signature(frame) for frame in iter_dump_frames(path)]
    assert fallback == sequential


def test_unrelated_neighbour_data_file_is_not_used(tmp_path):
    """A big shared folder must not donate elements to a small stray dump."""
    _write(
        tmp_path / "FeCoNi_relaxed.lmp",
        "LAMMPS data file\n\n97556 atoms\n3 atom types\n\nMasses\n\n1 55.845\n2 58.933194\n3 58.6934\n",
    )
    path = _write(tmp_path / "traj.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))

    suggestion = suggest_element_mapping(path)

    assert suggestion.mapping == {}
    assert suggestion.missing == (1, 2)
    with pytest.raises(ValueError, match=r"unresolved types: \[1, 2\]"):
        import_structures(path)


def test_matching_atom_count_data_file_is_used_even_with_another_stem(tmp_path):
    _write(
        tmp_path / "system.lmp",
        "LAMMPS data file\n\n2 atoms\n2 atom types\n\nMasses\n\n1 55.845\n2 58.933194\n",
    )
    path = _write(tmp_path / "traj.dump", _frame("id type x y z", ["1 1 0 0 0", "2 2 1 1 1"]))

    assert import_structures(path)[0].elements.tolist() == ["Fe", "Co"]


def test_last_mapping_round_trip(monkeypatch):
    import NepTrainKit.core.io.lammps_dump as module

    store: dict = {}
    monkeypatch.setattr(module, "_read_mapping_store", lambda: store)
    monkeypatch.setattr(module, "_write_mapping_store", lambda data: store.update(data))
    signature = ("id", "type", "x", "y", "z", "c_spin[1]", "c_spin[2]", "c_spin[3]", "c_spin[4]")
    overrides = module.ColumnOverrides(
        coordinate_names=("x", "y", "z"),
        force_names=(),
        spin_mode="magnitude_direction",
        spin_names=("c_spin[1]", "c_spin[2]", "c_spin[3]", "c_spin[4]"),
    )

    module.remember_dump_settings({1: "Fe", 2: "Co"}, signature=signature, overrides=overrides)

    saved = module.remembered_dump_settings((1, 2), signature=signature)
    assert saved is not None
    assert dict(saved.element_map) == {1: "Fe", 2: "Co"}
    assert saved.overrides == overrides
    partial = module.remembered_dump_settings((1, 3), signature=signature)
    assert partial is not None and dict(partial.element_map) == {1: "Fe"}
    assert module.remembered_dump_settings((3,), signature=signature) is None
    changed = module.remembered_dump_settings((1, 2), signature=("a", "b"))
    assert changed is not None and dict(changed.element_map) == {1: "Fe", 2: "Co"}
    assert changed.overrides is None


def test_column_overrides_define_custom_spin(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "custom.dump",
        _frame(
            "id element x y z c_mag[1] c_mag[2] c_mag[3] c_mag[4] c_pe",
            ["1 Fe 0 0 0 2.0 0.0 0.6 0.8 -5.0", "2 Fe 1 1 1 1.0 1.0 0.0 0.0 -5.1"],
        ),
    )
    overrides = ColumnOverrides(
        spin_mode="magnitude_direction",
        spin_names=("c_mag[1]", "c_mag[2]", "c_mag[3]", "c_mag[4]"),
        ignored_extras=frozenset({"dump.c_pe"}),
    )

    structure = import_structures(path, column_overrides=overrides)[0]

    np.testing.assert_allclose(
        structure.atomic_properties["spin"][0], [0.0, 1.2, 1.6], atol=1e-6
    )
    names = {prop["name"] for prop in structure.properties}
    assert "spin" in names
    assert "dump.c_pe" not in names
    assert "dump.c_mag[1]" not in names


def test_column_overrides_switch_the_coordinate_triplet(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "both.dump",
        _frame("id element x y z xs ys zs", ["1 Fe 9 9 9 0.1 0.2 0.3"]),
    )

    automatic = import_structures(path)[0]
    overridden = import_structures(
        path, column_overrides=ColumnOverrides(coordinate_names=("x", "y", "z"))
    )[0]

    np.testing.assert_allclose(automatic.positions, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(overridden.positions, [[9.0, 9.0, 9.0]])


def test_column_overrides_can_disable_forces(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "forces.dump",
        _frame("id element x y z fx fy fz", ["1 Fe 0 0 0 1 2 3"]),
    )

    structure = import_structures(path, column_overrides=ColumnOverrides(force_names=()))[0]

    assert "forces" not in {prop["name"] for prop in structure.properties}


def test_column_overrides_round_trip_through_dict():
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    overrides = ColumnOverrides(
        coordinate_names=("xs", "ys", "zs"),
        force_names=(),
        spin_mode="vector",
        spin_names=("a", "b", "c"),
        ignored_extras=frozenset({"dump.q"}),
    )

    assert ColumnOverrides.from_dict(overrides.to_dict()) == overrides


def test_loader_passes_column_overrides(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import NepTrainKit.core.io.lammps_dump as module

    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0"]))
    overrides = module.ColumnOverrides(force_names=())
    answer = SimpleNamespace(
        element_map={1: "Fe"},
        overrides=overrides,
        unwrap=False,
        keep_extra=True,
        strict=False,
        remember=False,
    )

    instance, _calls = _load_with(monkeypatch, path, answer)

    assert instance._import_options["column_overrides"] == overrides


def test_loader_persists_column_overrides_when_remembered(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import NepTrainKit.core.io.lammps_dump as module

    path = _write(tmp_path / "typed.dump", _frame("id type x y z", ["1 1 0 0 0"]))
    overrides = module.ColumnOverrides(spin_mode="off")
    captured: dict = {}
    monkeypatch.setattr(module, "remembered_dump_settings", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "remember_dump_settings",
        lambda mapping, **kwargs: captured.update(mapping=mapping, **kwargs),
    )
    answer = SimpleNamespace(
        element_map={1: "Fe"},
        overrides=overrides,
        unwrap=False,
        keep_extra=True,
        strict=False,
        remember=True,
    )

    _load_with(monkeypatch, path, answer)

    assert captured["overrides"] == overrides
    assert captured["signature"] == ("id", "type", "x", "y", "z")


def test_column_overrides_rename_extra_fields(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "extras.dump",
        _frame("id element x y z q c_pe", ["1 Fe 0 0 0 -1.0 -5.0"]),
    )

    structure = import_structures(
        path, column_overrides=ColumnOverrides(extra_names=(("dump.q", "charge"),))
    )[0]
    names = {prop["name"] for prop in structure.properties}

    assert "charge" in names
    assert "dump.q" not in names


def test_column_overrides_ignore_reserved_extra_renames(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "reserved.dump",
        _frame("id element x y z q", ["1 Fe 0 0 0 -1.0"]),
    )

    structure = import_structures(
        path, column_overrides=ColumnOverrides(extra_names=(("dump.q", "spin"),))
    )[0]
    names = {prop["name"] for prop in structure.properties}

    assert "dump.q" in names
    assert "spin" not in names


def test_column_overrides_extra_names_round_trip():
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    overrides = ColumnOverrides(extra_names=(("dump.q", "charge"), ("vel", "velocity")))

    assert ColumnOverrides.from_dict(overrides.to_dict()) == overrides


def test_column_overrides_merge_columns_into_one_field(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "group.dump",
        _frame(
            "id element x y z c_foo[1] c_foo[2] c_foo[3]",
            ["1 Fe 0 0 0 1.0 2.0 3.0"],
        ),
    )

    structure = import_structures(
        path,
        column_overrides=ColumnOverrides(
            custom_fields=(("foo", ("c_foo[1]", "c_foo[2]", "c_foo[3]")),)
        ),
    )[0]

    np.testing.assert_allclose(structure.atomic_properties["foo"], [[1.0, 2.0, 3.0]])
    prop = {item["name"]: item for item in structure.properties}["foo"]
    assert prop["type"] == "R"
    assert prop["count"] == 3
    names = {item["name"] for item in structure.properties}
    assert "dump.c_foo[1]" not in names


def test_custom_field_overlapping_a_role_is_ignored(tmp_path):
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    path = _write(
        tmp_path / "overlap.dump",
        _frame("id element x y z fx fy fz", ["1 Fe 0 0 0 1 2 3"]),
    )

    structure = import_structures(
        path,
        column_overrides=ColumnOverrides(custom_fields=(("bad", ("x", "y", "z")),)),
    )[0]

    assert "bad" not in {item["name"] for item in structure.properties}


def test_custom_field_round_trip():
    from NepTrainKit.core.io.lammps_dump import ColumnOverrides

    overrides = ColumnOverrides(custom_fields=(("foo", ("a", "b")), ("bar", ("c",))))

    assert ColumnOverrides.from_dict(overrides.to_dict()) == overrides
