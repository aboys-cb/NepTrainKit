"""Card-header exports publish complete datasets and preserve stream append."""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.io import atomic
from NepTrainKit.core.structure import Structure, write_structures_extxyz_atomic
from NepTrainKit.ui.views._card.card_group import CardGroup
from NepTrainKit.ui.views._card.workflow_fork import WorkflowFork
from NepTrainKit.ui.widgets.card_widget import MakeDataCard


@pytest.fixture(scope="module", autouse=True)
def application():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(params=["card", "group", "filtered_group", "fork"])
def exporter(request):
    if request.param == "card":
        card = MakeDataCard()
    elif request.param == "fork":
        card = WorkflowFork()
        card.merge_enabled = True
    else:
        card = CardGroup()
        disabled = MakeDataCard()
        disabled.state_checkbox.setChecked(False)
        disabled.result_dataset = [Atoms("H")]
        enabled = MakeDataCard()
        card.add_card(disabled)
        card.add_card(enabled)
    atoms = Atoms("Fe", cell=[3, 4, 5], pbc=[True, False, True])
    atoms.set_initial_magnetic_moments([[1., 2., 3.]])
    atoms.set_array("force_mag", np.array([[4., 5., 6.]]))
    atoms.set_array("group", np.array(["A"]))
    card.result_dataset = [atoms]
    if request.param in {"group", "filtered_group"}:
        enabled.result_dataset = [atoms]
        if request.param == "filtered_group":
            card.filter_card = MakeDataCard(card)
            card.filter_card.result_dataset = [atoms]
    yield card
    card.close()


@pytest.mark.parametrize("fail_at", ["write", "replace"])
@pytest.mark.parametrize("append", [False, True])
def test_card_header_export_failure_preserves_original(exporter, tmp_path, monkeypatch, fail_at, append):
    path = tmp_path / "existing.xyz"
    write(path, Atoms("Ne"))
    before = path.read_bytes()
    if fail_at == "replace":
        def fail_replace(*args):
            raise OSError("publish failed")
        monkeypatch.setattr(atomic.os, "replace", fail_replace)
    else:
        def fail_write(handle, *args, **kwargs):
            handle.write("partial frame")
            raise OSError("disk full")
        monkeypatch.setattr("NepTrainKit.ui.widgets.card_widget.ase_write", fail_write)
    with pytest.raises(OSError):
        exporter.write_result_dataset(path, append=append)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_card_header_export_replaces_and_preserves_canonical_fields(exporter, tmp_path):
    path = tmp_path / "existing.xyz"
    write(path, Atoms("Ne"))
    exporter.write_result_dataset(path)
    frames = read(path, ":")
    assert len(frames) == 1
    output = frames[0]
    assert output.get_chemical_formula() == "Fe"
    np.testing.assert_array_equal(output.arrays["spin"], [[1, 2, 3]])
    np.testing.assert_array_equal(output.arrays["force_mag"], [[4, 5, 6]])
    np.testing.assert_array_equal(output.arrays["group"], ["A"])
    assert "initial_magmoms" not in output.arrays
    assert "initial_magmoms" in exporter.result_dataset[0].arrays
    np.testing.assert_array_equal(output.cell, exporter.result_dataset[0].cell)
    np.testing.assert_array_equal(output.pbc, exporter.result_dataset[0].pbc)
    assert list(tmp_path.iterdir()) == [path]

    saved = tmp_path / "resaved.xyz"
    write_structures_extxyz_atomic(saved, Structure.read_multiple(path))
    reloaded = read(saved)
    for name in ("spin", "force_mag", "group"):
        np.testing.assert_array_equal(reloaded.arrays[name], output.arrays[name])


def test_card_export_appends_to_paths_and_supplied_workflow_streams(exporter, tmp_path):
    path = tmp_path / "append.xyz"
    write(path, Atoms("Ne"))
    exporter.write_result_dataset(path, append=True)
    assert [a.get_chemical_formula() for a in read(path, ":")] == ["Ne", "Fe"]
    with path.open("w", encoding="utf-8") as stream:
        exporter.write_result_dataset(stream, append=False)
        exporter.write_result_dataset(stream, append=True)
        assert not stream.closed
    assert [a.get_chemical_formula() for a in read(path, ":")] == ["Fe", "Fe"]


def test_group_failure_after_one_frame_preserves_original(tmp_path, monkeypatch):
    group = CardGroup()
    for symbol in ("H", "He"):
        child = MakeDataCard()
        child.result_dataset = [Atoms(symbol)]
        group.add_card(child)
    group.result_dataset = [atoms for child in group.card_list for atoms in child.result_dataset]
    path = tmp_path / "group.xyz"
    path.write_bytes(b"previous dataset\n")
    from NepTrainKit.ui.widgets import card_widget
    original_write = card_widget.ase_write
    def fail_second(handle, dataset, **kwargs):
        original_write(handle, dataset[:1], **kwargs)
        handle.write("partial second frame")
        raise OSError("second frame failed")
    monkeypatch.setattr(card_widget, "ase_write", fail_second)
    with pytest.raises(OSError):
        group.write_result_dataset(path)
    assert path.read_bytes() == b"previous dataset\n"
    assert list(tmp_path.iterdir()) == [path]
    group.close()
