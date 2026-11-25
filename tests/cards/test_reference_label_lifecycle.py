"""Changing a physical state cannot transfer the original DFT reference labels."""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from NepTrainKit.core.cards.defect import VacancyDefectOperation, VacancyDefectParams
from NepTrainKit.core.cards.lattice import (
    CellStrainOperation, CellStrainParams, PerturbOperation, PerturbParams,
    SuperCellOperation, SuperCellParams,
)
from NepTrainKit.core.cards.magnetism import MagneticMomentRotationOperation, MagneticMomentRotationParams
from NepTrainKit.core.cards.structure import GroupLabelOperation, GroupLabelParams
from NepTrainKit.core.io.importers import ase_atoms_to_structure
from NepTrainKit.core.structure import write_structures_extxyz_atomic
from NepTrainKit.ui.widgets.card_widget import write_card_dataset


def labelled_atoms():
    atoms = Atoms("Fe4", scaled_positions=[[0, 0, 0], [.25, .25, .25], [.5, .5, .5], [.75, .75, .75]],
                  cell=[[5, 0, 0], [.5, 5, 0], [.2, .3, 6]], pbc=[True, False, True])
    atoms.info.update(energy=-10., free_energy=-11., virial=np.arange(9.), stress=np.arange(6.),
                      Config_type="reference", source_note="keep me")
    for name in ("force", "forces", "force_mag", "mforce"):
        atoms.set_array(name, np.ones((4, 3)))
    atoms.set_array("charges", np.ones(4))
    atoms.set_array("spin", np.tile([0., 0., 2.], (4, 1)))
    atoms.set_initial_magnetic_moments(atoms.arrays["spin"])
    atoms.set_array("group", np.arange(4))
    atoms.set_array("custom", np.arange(4) + 10)
    return atoms


@pytest.mark.parametrize("operation,params", [
    (PerturbOperation(), PerturbParams(engine_type=1, max_num=2, use_seed=True, seed=12)),
    (CellStrainOperation(), CellStrainParams(axes="X", x_range=(1., 1., 1.))),
    (SuperCellOperation(), SuperCellParams(super_scale=(2, 1, 1))),
    (VacancyDefectOperation(), VacancyDefectParams(num_condition=1, max_structures=1, use_seed=True)),
    (MagneticMomentRotationOperation(), MagneticMomentRotationParams(num_structures=2, disturb_magnitude=False, use_seed=True)),
])
def test_changed_geometry_atom_count_and_spin_drop_only_dependent_labels(operation, params):
    seed = labelled_atoms()
    outputs = operation.run_structure(seed, params)
    assert outputs
    for output in outputs:
        assert not {"energy", "free_energy", "virial", "stress"}.intersection(output.info)
        assert not {"force", "forces", "force_mag", "mforce", "charges"}.intersection(output.arrays)
        assert output.info["source_note"] == "keep me"
        assert output.info["Config_type"].startswith("reference")
        assert {"spin", "initial_magmoms", "group", "custom"}.issubset(output.arrays)
        np.testing.assert_equal(output.arrays["custom"], output.arrays["group"] + 10)
        np.testing.assert_allclose(np.linalg.norm(output.arrays["spin"], axis=1), 2.)
        np.testing.assert_equal(output.pbc, seed.pbc)
    assert seed.info["energy"] == -10.
    assert "force_mag" in seed.arrays


@pytest.mark.parametrize("operation,params", [
    (PerturbOperation(), PerturbParams(engine_type=1, max_num=1, max_distance=0.)),
    (CellStrainOperation(), CellStrainParams(axes="X", x_range=(0., 0., 1.))),
    (SuperCellOperation(), SuperCellParams(super_scale=(1, 1, 1))),
    (GroupLabelOperation(), GroupLabelParams(overwrite=True)),
])
def test_noop_and_group_only_changes_keep_reference_labels(operation, params):
    seed = labelled_atoms()
    # Exact fractional representations avoid introducing a rounding displacement.
    seed.set_cell([4, 4, 4], scale_atoms=True)
    output = operation.run_structure(seed, params)[0]
    assert output.info["energy"] == -10.
    np.testing.assert_equal(output.info["virial"], seed.info["virial"])
    np.testing.assert_equal(output.arrays["force_mag"], seed.arrays["force_mag"])


def test_changed_labels_stay_absent_in_card_export_handoff_and_resave(tmp_path):
    output = PerturbOperation().run_structure(labelled_atoms(),
        PerturbParams(engine_type=1, max_num=1, use_seed=True, seed=12))[0]
    direct = tmp_path / "card.xyz"
    write_card_dataset(direct, [output])
    handed_off = ase_atoms_to_structure(output)
    resaved = tmp_path / "display.xyz"
    write_structures_extxyz_atomic(resaved, [handed_off])
    for path in (direct, resaved):
        loaded = read(path)
        assert not {"virial", "stress"}.intersection(loaded.info)
        assert not {"force", "forces", "force_mag", "mforce"}.intersection(loaded.arrays)
        assert "energy" not in (getattr(loaded.calc, "results", {}) or {})
        assert "spin" in loaded.arrays
        np.testing.assert_allclose(loaded.positions, output.positions, rtol=0., atol=1e-6)
