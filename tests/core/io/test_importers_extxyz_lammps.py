from __future__ import annotations

import numpy as np
import pytest

from NepTrainKit.core.io.importers import import_structures, is_parseable


def test_extxyz_importer_loads_lattice_energy_positions_and_forces(tmp_path):
    path = tmp_path / "train.xyz"
    path.write_text(
        "\n".join(
            [
                "2",
                'Lattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.5 Config_type=test pbc="T T T"',
                "H 0 0 0 0.1 0.2 0.3",
                "He 1 1 1 -0.1 -0.2 -0.3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structures = import_structures(path)

    assert len(structures) == 1
    structure = structures[0]
    np.testing.assert_allclose(structure.lattice, np.diag([3.0, 3.0, 3.0]))
    assert structure.additional_fields["Config_type"] == "test"
    assert structure.energy == -1.5
    assert structure.elements.tolist() == ["H", "He"]
    np.testing.assert_allclose(structure.positions[1], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(structure.forces[0], [0.1, 0.2, 0.3])


def test_matching_invalid_extxyz_raises_value_error(tmp_path):
    path = tmp_path / "broken.xyz"
    path.write_text("not an xyz file\n", encoding="utf-8")

    assert is_parseable(path)
    with pytest.raises(ValueError, match="Failed to import structures"):
        import_structures(path)


def test_unmatched_file_returns_empty_list(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text\n", encoding="utf-8")

    assert not is_parseable(path)
    assert import_structures(path) == []


@pytest.mark.parametrize("content", ["", "\n\n  \t\n"])
def test_empty_file_returns_empty_list(tmp_path, content):
    path = tmp_path / "empty.xyz"
    path.write_text(content, encoding="utf-8")

    assert is_parseable(path)
    assert import_structures(path) == []


def test_lammps_dump_importer_loads_scaled_coordinates_forces_and_elements(tmp_path):
    path = tmp_path / "traj.dump"
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "7",
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp pp pp",
                "0 10",
                "0 20",
                "0 30",
                "ITEM: ATOMS id element xs ys zs fx fy fz",
                "1 H 0.1 0.2 0.3 1 2 3",
                "2 He 0.5 0.5 0.5 -1 -2 -3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structures = import_structures(path)

    assert len(structures) == 1
    structure = structures[0]
    assert structure.additional_fields["Config_type"] == "LAMMPS_7"
    assert structure.atomic_properties["species"].tolist() == ["H", "He"]
    np.testing.assert_allclose(structure.lattice, np.diag([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(structure.positions, [[1.0, 4.0, 9.0], [5.0, 10.0, 15.0]])
    np.testing.assert_allclose(structure.forces, [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])


def test_matching_invalid_lammps_dump_raises_value_error(tmp_path):
    path = tmp_path / "broken.dump"
    path.write_text("ITEM: TIMESTEP\n0\n", encoding="utf-8")

    assert is_parseable(path)
    with pytest.raises(ValueError, match="Failed to import structures"):
        import_structures(path)
