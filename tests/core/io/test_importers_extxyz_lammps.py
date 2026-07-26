from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from NepTrainKit.core.io.importers import (
    ase_atoms_to_structure,
    import_structures,
    is_parseable,
)


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


def test_ase_trajectory_without_calculator_preserves_structure_and_metadata(tmp_path):
    path = tmp_path / "plain structure.traj"
    atoms = Atoms(
        "FeO",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.5, 2.0]],
        cell=np.diag([3.0, 4.0, 5.0]),
        pbc=[True, False, True],
        info={"Config_type": "plain", "provenance": "原始 轨迹"},
    )
    atoms.set_tags([3, 7])
    atoms.set_initial_magnetic_moments([2.1, -0.2])
    ase_write(path, atoms)

    structure = import_structures(path)[0]

    assert structure.additional_fields["Config_type"] == "plain"
    assert structure.additional_fields["provenance"] == "原始 轨迹"
    assert structure.additional_fields["pbc"] == "T F T"
    assert "energy" not in structure.additional_fields
    np.testing.assert_allclose(structure.lattice, np.diag([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(structure.positions[1], [1.0, 1.5, 2.0])
    np.testing.assert_array_equal(structure.atomic_properties["tags"], [3, 7])
    np.testing.assert_allclose(
        structure.atomic_properties["initial_magmoms"],
        [2.1, -0.2],
    )


def test_in_memory_ase_conversion_preserves_scientific_fields():
    atoms = Atoms(
        "FeO",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.5, 2.0]],
        cell=np.diag([3.0, 4.0, 5.0]),
        pbc=[True, True, False],
        info={
            "Config_type": "card-output",
            "provenance": "Make Dataset",
            "energy": -3.25,
            "virial": np.arange(9, dtype=float),
        },
    )
    atoms.set_array(
        "spin",
        np.asarray([[0.0, 0.0, 2.1], [0.0, 0.0, -0.2]]),
    )
    atoms.set_array(
        "forces",
        np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
    )

    structure = ase_atoms_to_structure(atoms)

    assert structure.additional_fields["Config_type"] == "card-output"
    assert structure.additional_fields["provenance"] == "Make Dataset"
    assert structure.additional_fields["pbc"] == "T T F"
    assert structure.energy == -3.25
    np.testing.assert_allclose(structure.lattice, np.diag([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(structure.positions, atoms.positions)
    np.testing.assert_allclose(structure.atomic_properties["spin"], atoms.arrays["spin"])
    np.testing.assert_allclose(structure.forces, atoms.arrays["forces"])
    np.testing.assert_allclose(
        structure.additional_fields["virial"],
        np.arange(9, dtype=float),
    )


def test_ase_trajectory_preserves_calculator_labels_and_tensor_order(tmp_path):
    path = tmp_path / "labeled.traj"
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=True,
    )
    forces = np.arange(6, dtype=float).reshape(2, 3)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-4.5,
        forces=forces,
        stress=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    ase_write(path, atoms)

    structure = import_structures(path)[0]

    assert structure.additional_fields["energy"] == -4.5
    np.testing.assert_allclose(structure.atomic_properties["forces"], forces)
    np.testing.assert_allclose(
        structure.additional_fields["stress"],
        [1.0, 6.0, 5.0, 6.0, 2.0, 4.0, 5.0, 4.0, 3.0],
    )


def test_corrupt_ase_trajectory_reports_the_read_error(tmp_path):
    path = tmp_path / "broken.traj"
    path.write_bytes(b"not an ASE trajectory")

    with pytest.raises(ValueError, match="Failed to read ASE trajectory"):
        import_structures(path)


def test_n2p2_import_preserves_units_labels_and_per_atom_fields(tmp_path):
    path = tmp_path / "input.data"
    path.write_text(
        "\n".join(
            [
                "begin",
                "comment water/source",
                "lattice 4 0 0",
                "lattice 0 5 0",
                "lattice 0 0 6",
                "atom 0 0 0 H 0.25 -0.5 1 2 3",
                "atom 1 2 3 O -0.25 -1.0 -1 -2 -3",
                "energy -2.0",
                "charge 0.0",
                "end",
                "",
            ]
        ),
        encoding="utf8",
    )

    structure = import_structures(path)[0]

    bohr_to_angstrom = 1.0 / 1.8897261328
    hartree_to_ev = 1.0 / 0.0367493254
    np.testing.assert_allclose(
        structure.lattice,
        np.diag([4.0, 5.0, 6.0]) * bohr_to_angstrom,
    )
    np.testing.assert_allclose(
        structure.positions[1],
        np.asarray([1.0, 2.0, 3.0]) * bohr_to_angstrom,
    )
    np.testing.assert_allclose(
        structure.forces[0],
        np.asarray([1.0, 2.0, 3.0]) * hartree_to_ev / bohr_to_angstrom,
    )
    np.testing.assert_allclose(structure.atomic_properties["charge"], [0.25, -0.25])
    np.testing.assert_allclose(
        structure.atomic_properties["atomic_energy"],
        np.asarray([-0.5, -1.0]) * hartree_to_ev,
    )
    assert structure.energy == pytest.approx(-2.0 * hartree_to_ev)
    assert structure.additional_fields["Config_type"] == "water/source"
    assert structure.additional_fields["pbc"] == "T T T"


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            "begin\nlattice 1 0 0\natom 0 0 0 H 0 0 0 0 0\nend\n",
            "zero or three lattice rows",
        ),
        (
            "begin\natom 0 0 0 H 0 0 bad 0 0\nend\n",
            "non-numeric atom field",
        ),
        (
            "begin\natom 0 0 0 H 0 0 0 0 0\n",
            "missing its closing 'end'",
        ),
    ],
)
def test_n2p2_import_rejects_partial_or_corrupt_blocks(tmp_path, body, error):
    path = tmp_path / "broken.cfg"
    path.write_text(body, encoding="utf8")

    with pytest.raises(ValueError, match=error):
        import_structures(path)


def test_cp2k_single_point_import_preserves_units_cell_and_labels(tmp_path):
    path = tmp_path / "cp2k.out"
    path.write_text(
        "\n".join(
            [
                " CP2K| version string",
                " CELL| Vector a [angstrom]: 3.0 0.0 0.0",
                " CELL| Vector b [angstrom]: 0.0 4.0 0.0",
                " CELL| Vector c [angstrom]: 0.0 0.0 5.0",
                " MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM",
                " Atom Kind Element Z X Y Z Z(eff) Mass",
                " 1 1 H 1 0.0 0.0 0.0 1.0 1.0",
                " 2 1 H 1 1.0 2.0 3.0 1.0 1.0",
                "",
                " ATOMIC FORCES in [a.u.]",
                " # Atom Kind Element X Y Z",
                " 1 1 H 1.0 0.0 0.0",
                " 2 1 H -1.0 0.0 0.0",
                " SUM OF ATOMIC FORCES",
                " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -2.0",
                " STRESS| Analytical stress tensor [GPa]",
                " STRESS| x y z",
                " STRESS| x 1.0 2.0 3.0",
                " STRESS| y 4.0 5.0 6.0",
                " STRESS| z 7.0 8.0 9.0",
                "",
            ]
        ),
        encoding="utf8",
    )

    structure = import_structures(path)[0]

    np.testing.assert_allclose(structure.lattice, np.diag([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(structure.positions[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        structure.forces[:, 0],
        [27.211386245988 / 0.52917721067, -27.211386245988 / 0.52917721067],
    )
    assert structure.energy == pytest.approx(-2.0 * 27.211386245988)
    np.testing.assert_allclose(
        structure.additional_fields["stress"],
        np.arange(1.0, 10.0) / 160.21766208,
    )
    assert structure.additional_fields["pbc"] == "T T T"


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            "\n".join(
                [
                    "CP2K|",
                    "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM",
                    "1 1 H 1 0 0 0",
                    "",
                ]
            ),
            "missing complete CELL vectors",
        ),
        (
            "\n".join(
                [
                    "CP2K|",
                    "CELL| Vector a [angstrom]: 3 0 0",
                    "CELL| Vector b [angstrom]: 0 3 0",
                    "CELL| Vector c [angstrom]: 0 0 3",
                    "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM",
                    "1 1 H 1 0 0 0",
                    "",
                    "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM",
                    "1 1 H 1 0 0 0",
                    "",
                ]
            ),
            "multiple coordinate blocks",
        ),
    ],
)
def test_cp2k_import_rejects_ambiguous_or_incomplete_outputs(
    tmp_path, body, error
):
    path = tmp_path / "broken.log"
    path.write_text(body, encoding="utf8")

    with pytest.raises(ValueError, match=error):
        import_structures(path)


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


def test_lammps_dump_numeric_types_require_explicit_element_mapping(tmp_path):
    path = tmp_path / "typed.dump"
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "0",
                "ITEM: NUMBER OF ATOMS",
                "2",
                "ITEM: BOX BOUNDS pp ff pp",
                "0 4",
                "0 5",
                "0 6",
                "ITEM: ATOMS id type x y z",
                "1 1 0 0 0",
                "2 2 2 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved types: \\[1, 2\\]"):
        import_structures(path)

    structure = import_structures(path, element_map={1: "H", 2: "He"})[0]
    assert structure.elements.tolist() == ["H", "He"]
    assert structure.additional_fields["pbc"] == "T F T"
    np.testing.assert_allclose(structure.positions, [[0, 0, 0], [2, 3, 4]])


def test_lammps_dump_triclinic_cartesian_coordinates_preserve_cell_and_positions(tmp_path):
    path = tmp_path / "triclinic.dump"
    path.write_text(
        "\n".join(
            [
                "ITEM: TIMESTEP",
                "3",
                "ITEM: NUMBER OF ATOMS",
                "1",
                "ITEM: BOX BOUNDS xy xz yz pp pp pp",
                "0 7 2",
                "0 6 1",
                "0 4 -1",
                "ITEM: ATOMS id element x y z",
                "1 Si 2.5 2.5 2.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    structure = import_structures(path)[0]

    np.testing.assert_allclose(
        structure.lattice,
        [[4, 0, 0], [2, 5, 0], [1, -1, 4]],
    )
    np.testing.assert_allclose(structure.positions, [[2.5, 1.5, 2.0]])


@pytest.mark.parametrize(
    ("atoms_header", "atom_rows", "message"),
    [
        (
            "ITEM: ATOMS id element",
            ["1 H"],
            "must contain one complete coordinate triplet",
        ),
        (
            "ITEM: ATOMS id element x y z",
            ["1 H 0 0 0"],
            "expected 2 atom rows, found 1",
        ),
        (
            "ITEM: ATOMS id x y z",
            ["1 0 0 0", "2 1 1 1"],
            "must contain an element or type column",
        ),
    ],
)
def test_lammps_dump_rejects_incomplete_scientific_contract(
    tmp_path, atoms_header, atom_rows, message
):
    path = tmp_path / "incomplete.dump"
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
                atoms_header,
                *atom_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        import_structures(path)


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
