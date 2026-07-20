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


def test_lammps_dump_importer_reconstructs_spin_vectors(tmp_path):
    path = tmp_path / "spin.dump"
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "0",
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp pp pp",
                "0 4",
                "0 4",
                "0 4",
                "ITEM: ATOMS id element x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]",
                "1 Fe 0 0 0 2.0 0.0 0.6 0.8 90 91 92",
                "2 Fe 2 2 2 1.5 -1.0 0.0 0.0 93 94 95",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structure = import_structures(path)[0]

    np.testing.assert_allclose(
        structure.atomic_properties["spin"],
        [[0.0, 1.2, 1.6], [-1.5, 0.0, 0.0]],
    )
    assert {prop["name"]: prop for prop in structure.properties}["spin"] == {
        "name": "spin",
        "type": "R",
        "count": 3,
    }


def test_matching_invalid_lammps_dump_raises_value_error(tmp_path):
    path = tmp_path / "broken.dump"
    path.write_text("ITEM: TIMESTEP\n0\n", encoding="utf-8")

    assert is_parseable(path)
    with pytest.raises(ValueError, match="Failed to import structures"):
        import_structures(path)


def test_outcar_importer_uses_potcar_species_when_vrhfin_is_absent(tmp_path):
    path = tmp_path / "OUTCAR (1)"
    path.write_text(
        "\n".join(
            [
                " POTCAR:    PAW_PBE Cl 06Sep2000",
                " POTCAR:    PAW_PBE Li_sv 10Sep2004",
                " POTCAR:    PAW_PBE U 06Sep2000",
                "   ions per type =               1   1   1",
                "      direct lattice vectors                 reciprocal lattice vectors",
                "     3.0 0.0 0.0    0 0 0",
                "     0.0 3.0 0.0    0 0 0",
                "     0.0 0.0 3.0    0 0 0",
                " POSITION                                       TOTAL-FORCE (eV/Angst)",
                " -----------------------------------------------------------------------------------",
                "   0.0 0.0 0.0   0.1 0.2 0.3",
                "   1.0 1.0 1.0  -0.1 -0.2 -0.3",
                "   2.0 2.0 2.0   0.0 0.0 0.0",
                " -----------------------------------------------------------------------------------",
                "  free  energy   TOTEN  =      -12.50000000 eV",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structures = import_structures(path)

    assert len(structures) == 1
    structure = structures[0]
    assert structure.atomic_properties["species"].tolist() == ["Cl", "Li", "U"]
    assert structure.energy == -12.5


def test_outcar_importer_falls_back_to_position_block_without_forces(tmp_path):
    path = tmp_path / "sample.outcar"
    path.write_text(
        "\n".join(
            [
                " POTCAR:    PAW_PBE Cl 06Sep2000",
                " POTCAR:    PAW_PBE Li_sv 10Sep2004",
                "   ions per type =               1   1",
                "      direct lattice vectors                 reciprocal lattice vectors",
                "     2.0 0.0 0.0    0 0 0",
                "     0.0 2.0 0.0    0 0 0",
                "     0.0 0.0 2.0    0 0 0",
                " position of ions in fractional coordinates (direct lattice)",
                "   0.25 0.25 0.25",
                "   0.75 0.75 0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structures = import_structures(path)

    assert len(structures) == 1
    structure = structures[0]
    assert structure.atomic_properties["species"].tolist() == ["Cl", "Li"]
    assert "forces" not in structure.atomic_properties
    np.testing.assert_allclose(structure.positions, [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])


def test_xdatcar_importer_loads_all_configurations(tmp_path):
    path = tmp_path / "XDATCAR"
    path.write_text(
        "\n".join(
            [
                "sample",
                "1.0",
                "2 0 0",
                "0 2 0",
                "0 0 2",
                "Cl Li",
                "1 1",
                "Direct configuration=     1",
                "0.0 0.0 0.0",
                "0.5 0.5 0.5",
                "Direct configuration=     2",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structures = import_structures(path)

    assert len(structures) == 2
    assert structures[0].atomic_properties["species"].tolist() == ["Cl", "Li"]
    assert structures[1].additional_fields["Config_type"] == "XDATCAR_2"
    np.testing.assert_allclose(structures[1].positions, [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])


def test_xdatcar_importer_keeps_repeated_header_compatibility(tmp_path):
    path = tmp_path / "XDATCAR"
    path.write_text(
        "\n".join(
            [
                "frame one",
                "1.0",
                "2 0 0",
                "0 2 0",
                "0 0 2",
                "Cl",
                "1",
                "Direct configuration=     1",
                "0.25 0.25 0.25",
                "frame two",
                "1.0",
                "4 0 0",
                "0 4 0",
                "0 0 4",
                "Cl",
                "1",
                "Direct configuration=     2",
                "0.25 0.25 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structures = import_structures(path)

    assert len(structures) == 2
    np.testing.assert_allclose(structures[0].positions, [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(structures[1].positions, [[1.0, 1.0, 1.0]])
