"""Regressions for stale labels, bounded planning, and truthful spin sampling."""

from dataclasses import replace
from io import StringIO

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrainKit.core.cards.alloy import (
    CompositionGradientOperation,
    CompositionGradientParams,
    ConditionalReplaceOperation,
    ConditionalReplaceParams,
    FiniteCellAlloyOccupancyOperation,
    FiniteCellAlloyOccupancyParams,
    RandomOccupancyOperation,
    RandomOccupancyParams,
)
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.lattice import SuperCellOperation, SuperCellParams
from NepTrainKit.core.cards.magnetism import SmallAngleSpinTiltOperation, SmallAngleSpinTiltParams
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export


@pytest.mark.parametrize("operation,params", [
    (ConditionalReplaceOperation(), ConditionalReplaceParams(target="Fe", replacements="Co:1")),
    (CompositionGradientOperation(), CompositionGradientParams(
        elements="Fe,Co", start_composition="Fe:1", end_composition="Co:1", bins=2, use_seed=True)),
    (RandomOccupancyOperation(), RandomOccupancyParams(source="Manual", manual="Co:1", use_seed=True)),
    (FiniteCellAlloyOccupancyOperation(), FiniteCellAlloyOccupancyParams(
        site_rules='{"all":{"elements":["Co"],"mode":"fixed_fraction","composition":{"Co":1}}}')),
])
def test_species_changes_drop_reference_labels_but_keep_structure_and_provenance(operation, params):
    seed = Atoms("Fe2", scaled_positions=[[0, 0, 0], [.6, .2, .4]],
                 cell=[[5, 0, 0], [1, 4, 0], [.5, .5, 6]], pbc=[True, False, True])
    seed.info.update(energy=-1., virial=np.eye(3), source_note="original", Config_type="seed")
    for name in ("force", "forces", "force_mag", "mforce"):
        seed.set_array(name, np.ones((2, 3)))
    seed.set_array("group", np.array(["A", "B"]))
    result = operation.run_structure(seed, params)[0]
    assert result.get_chemical_symbols() != seed.get_chemical_symbols()
    assert not {"energy", "virial"}.intersection(result.info)
    assert not {"force", "forces", "force_mag", "mforce"}.intersection(result.arrays)
    np.testing.assert_array_equal(result.cell, seed.cell)
    np.testing.assert_array_equal(result.pbc, seed.pbc)
    np.testing.assert_array_equal(result.positions, seed.positions)
    np.testing.assert_array_equal(result.arrays["group"], seed.arrays["group"])
    assert result.info["source_note"] == "original"
    assert result.info["Config_type"].startswith("seed|")
    stream = StringIO()
    write(stream, prepare_magnetic_extxyz_export(result), format="extxyz")
    stream.seek(0)
    loaded = read(stream, format="extxyz")
    assert not {"force_mag", "mforce"}.intersection(loaded.arrays)
    assert seed.info["energy"] == -1.
    assert "mforce" in seed.arrays


def test_noop_occupancy_keeps_reference_labels():
    seed = Atoms("Fe2")
    seed.info["energy"] = -1.
    seed.set_array("mforce", np.ones((2, 3)))
    result = RandomOccupancyOperation().run_structure(
        seed, RandomOccupancyParams(source="Manual", manual="Fe:1", use_seed=True))[0]
    assert result.info["energy"] == -1.
    np.testing.assert_array_equal(result.arrays["mforce"], seed.arrays["mforce"])


@pytest.mark.parametrize("mode", ["scale", "cell"])
def test_supercell_rejects_huge_enumeration_before_materializing_candidates(mode, monkeypatch):
    operation = SuperCellOperation()
    seed = Atoms("Fe", cell=np.eye(3), pbc=True)
    params = SuperCellParams(mode=mode, output_mode="enumerate",
                            super_scale=(999, 999, 999), target_cell=(999, 999, 999))
    def unexpected(*args):
        pytest.fail("over-limit enumeration must fail before candidate deduplication")
    monkeypatch.setattr(operation, "_dedupe_factors", unexpected)
    with pytest.raises(CardOperationError, match="997002999"):
        operation.plan_factors(seed, params)


def test_supercell_enumeration_limit_and_fixed_axes():
    operation = SuperCellOperation()
    seed = Atoms("Fe", cell=np.eye(3), pbc=True)
    params = SuperCellParams(output_mode="enumerate", super_scale=(10, 10, 10))
    assert len(operation.plan_factors(seed, params)) == 1000
    with pytest.raises(CardOperationError):
        operation.plan_factors(seed, replace(params, super_scale=(1001, 1, 1)))
    fixed = replace(params, super_scale=(999, 10, 10),
                    fixed_axis_flags=(True, False, False), fixed_axis_scale=(2, 1, 1))
    assert len(operation.plan_factors(seed, fixed)) == 100
    single = replace(params, output_mode="single", super_scale=(999, 999, 999))
    assert operation.plan_factors(seed, single) == [(999, 999, 999)]


@pytest.mark.parametrize("angles", ["", "typo", "0,-5", "1,typo", "nan", "inf", "1,-2"])
def test_spin_tilt_rejects_invalid_angle_lists(angles):
    seed = Atoms("Fe2")
    seed.set_initial_magnetic_moments([[0, 0, 2], [0, 0, -2]])
    before = seed.get_initial_magnetic_moments().copy()
    with pytest.raises(CardOperationError, match="Small-angle Spin Tilt"):
        SmallAngleSpinTiltOperation().run_structure(seed, SmallAngleSpinTiltParams(angle_list=angles))
    np.testing.assert_array_equal(seed.get_initial_magnetic_moments(), before)


def test_spin_group_canting_requires_distinct_groups_and_preserves_magnitudes():
    seed = Atoms("Fe2")
    seed.set_initial_magnetic_moments([[0, 0, 2], [0, 0, 2]])
    seed.set_array("group", np.array(["A", "B"]))
    params = SmallAngleSpinTiltParams(canting_mode="Group pair canting", group_a="A", group_b="A",
                                     angle_list="10,10", include_reference=False)
    operation = SmallAngleSpinTiltOperation()
    with pytest.raises(CardOperationError, match="must be different"):
        operation.run_structure(seed, params)
    results = operation.run_structure(seed, replace(params, group_b="B"))
    assert len(results) == 1
    moments = results[0].get_initial_magnetic_moments()
    np.testing.assert_allclose(np.linalg.norm(moments, axis=1), [2, 2])
    angle = np.degrees(np.arccos(np.clip(np.dot(moments[0], moments[1]) / 4., -1., 1.)))
    assert angle == pytest.approx(10.)
