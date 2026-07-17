import sys
from pathlib import Path

from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_make_fcc_dump import make_fcc_random_alloy, write_lammps_dump


def test_make_fcc_random_alloy_writes_dump_and_extxyz(tmp_path):
    atoms = make_fcc_random_alloy(["Ni", "Co", "Cr"], size=2, lattice=3.6, seed=7)
    dump_path = tmp_path / "fcc.dump"
    xyz_path = tmp_path / "fcc.xyz"

    write_lammps_dump(dump_path, atoms, ["Ni", "Co", "Cr"])
    atoms.write(xyz_path, format="extxyz")

    text = dump_path.read_text(encoding="utf-8")
    assert "ITEM: ATOMS id type x y z" in text
    assert "ITEM: NUMBER OF ATOMS\n32\n" in text
    assert len(read(xyz_path, index=0)) == 32
