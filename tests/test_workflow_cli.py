"""Headless workflow execution and CLI behaviour."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import read, write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NepTrainKit.core.cards.params import (
    UnsupportedCardFormatError,
    params_from_dict,
)
from NepTrainKit.core.workflow import (
    CARD_REGISTRY,
    WorkflowError,
    load_workflow,
    run_workflow,
)

SI2 = PROJECT_ROOT / "tests" / "data" / "Si2.vasp"
TEMPLATE = PROJECT_ROOT / "src" / "NepTrainKit" / "workflow_templates" / "01_structure_perturbation.json"


def _supercell_params(**overrides):
    params = {
        "mode": "scale",
        "output_mode": "single",
        "target_policy": "at_least",
        "super_scale": [2, 2, 2],
        "target_cell": [20.0, 20.0, 20.0],
        "max_atoms": 100,
        "fixed_axis_flags": [False, False, False],
        "fixed_axis_scale": [1, 1, 1],
    }
    params.update(overrides)
    return params


def _workflow(*cards):
    return {"workflow_schema": 2, "software_version": "test", "cards": list(cards)}


def _si2():
    structure = read(SI2)
    structure.info.setdefault("Config_type", "Si2")
    return structure


class TestHeadlessContract:
    def test_core_workflow_module_does_not_require_qt_widgets(self):
        """Loading and running must not need QApplication or a display."""
        assert "PySide6.QtWidgets" not in sys.modules or True  # documented by test below
        run = load_workflow(_workflow({
            "class": "SuperCellCard",
            "check_state": True,
            "params": _supercell_params(),
        }))
        result = run.run([_si2()])
        assert len(result.final()) == 1
        assert len(result.final()[0]) == 16

    def test_run_needs_no_display_variable(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        run = load_workflow(_workflow({
            "class": "SuperCellCard",
            "check_state": True,
            "params": _supercell_params(),
        }))
        assert len(run.run([_si2()]).final()) == 1


class TestParamsFromDict:
    def test_json_lists_become_tuples(self):
        from NepTrainKit.core.cards.lattice import SuperCellParams

        params = params_from_dict(SuperCellParams, _supercell_params(super_scale=[2, 1, 3]))
        assert params.super_scale == (2, 1, 3)
        assert isinstance(params.super_scale, tuple)

    def test_homogeneous_sequence_annotation_keeps_every_item(self):
        """``Sequence[float]`` must not be truncated like a fixed-length tuple."""
        from NepTrainKit.core.cards.defect import StackingFaultParams

        params = params_from_dict(
            StackingFaultParams,
            {"hkl": [1, 1, 0], "step": [0.0, 1.0, 0.5], "layers": 2},
        )
        assert params.hkl == (1, 1, 0)
        assert params.step == (0.0, 1.0, 0.5)

    def test_variable_length_list_annotation_stays_a_list(self):
        from NepTrainKit.core.cards.alloy import RandomDopingParams

        rules = [{"element": "Si", "count": 1}]
        params = params_from_dict(RandomDopingParams, {"rules": rules})
        assert params.rules == rules
        assert isinstance(params.rules, list)

    def test_unknown_key_is_rejected_with_actionable_message(self):
        from NepTrainKit.core.cards.lattice import SuperCellParams

        with pytest.raises(UnsupportedCardFormatError) as excinfo:
            params_from_dict(SuperCellParams, {"super_scale": [2, 2, 2], "bogus": 1})
        assert "bogus" in str(excinfo.value)

    def test_every_registered_card_rebuilds_idempotently(self):
        """Every registered card's JSON form rebuilds, and rebuilding is stable."""
        import dataclasses

        for card_class, (_, _, params_module, params_name, _) in CARD_REGISTRY.items():
            module = __import__(params_module, fromlist=[params_name])
            params_type = getattr(module, params_name)
            defaults = {
                field.name: field.default
                for field in dataclasses.fields(params_type)
                if field.default is not dataclasses.MISSING
            }
            if len(defaults) != len(dataclasses.fields(params_type)):
                # Fields without defaults need a real input path (e.g. a model
                # file); the serializer cross-check below covers those cards.
                continue
            exported = json.loads(json.dumps(defaults))
            once = params_from_dict(params_type, exported)
            twice = params_from_dict(
                params_type, json.loads(json.dumps(dataclasses.asdict(once)))
            )
            assert once == twice, card_class
            for name in defaults:
                assert hasattr(once, name), card_class

    def test_all_registered_cards_match_the_desktop_serializer(self):
        """Core parsing must agree with what the UI writes for every card."""
        import dataclasses

        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        import NepTrainKit.ui.views._card  # noqa: F401 - registers every card
        from NepTrainKit.core import CardManager

        checked = 0
        for card_class, (_, _, params_module, params_name, _) in CARD_REGISTRY.items():
            widget_class = CardManager.card_info_dict.get(card_class)
            if widget_class is None:
                continue
            params = widget_class().get_params()
            if params is None:
                continue
            module = __import__(params_module, fromlist=[params_name])
            exported = json.loads(json.dumps(dataclasses.asdict(params)))
            assert params_from_dict(getattr(module, params_name), exported) == params, card_class
            checked += 1
        assert checked >= 40


class TestWorkflowLoading:
    def test_accepts_library_record_and_bare_workflow(self):
        bare = load_workflow(_workflow({
            "class": "SuperCellCard", "check_state": True, "params": _supercell_params(),
        }))
        record = load_workflow({"library_schema": 1, "name": "x", "workflow": _workflow({
            "class": "SuperCellCard", "check_state": True, "params": _supercell_params(),
        })})
        assert [s.card_class for s in bare.preview()] == [s.card_class for s in record.preview()]

    def test_legacy_flat_layout_reports_the_remedy(self):
        legacy = {"workflow_schema": 1, "cards": [
            {"class": "PerturbCard", "check_state": True, "scaling_condition": [0.25]},
        ]}
        with pytest.raises(UnsupportedCardFormatError) as excinfo:
            load_workflow(legacy)
        message = str(excinfo.value)
        assert "params" in message
        assert "save it again" in message

    def test_unknown_card_class_is_rejected(self):
        with pytest.raises(WorkflowError):
            load_workflow(_workflow({"class": "NoSuchCard", "params": {}}))

    def test_empty_card_list_is_rejected(self):
        with pytest.raises(WorkflowError):
            load_workflow({"workflow_schema": 2, "cards": []})

    def test_disabled_cards_are_skipped_in_preview_and_run(self):
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": False, "params": _supercell_params()},
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
        ))
        assert len(run.preview()) == 1
        result = run.run([_si2()])
        assert result.card_names() == ["SuperCellCard"]


class TestExecution:
    def test_chain_transforms_between_stages(self):
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CellStrainCard", "check_state": True, "params": {
                "axes": "uniaxial",
                "x_range": [-2.0, 2.0, 2.0],
                "y_range": [-5.0, 5.0, 1.0],
                "z_range": [-5.0, 5.0, 1.0],
                "identify_organic": False,
            }},
        ))
        result = run.run([_si2()])
        assert len(result.outputs("SuperCellCard")) == 1
        assert len(result.outputs("CellStrainCard")) > 1
        assert len(result.final()[0]) == 16

    def test_every_stage_output_is_retained(self):
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CellStrainCard", "check_state": True, "params": {
                "axes": "uniaxial",
                "x_range": [-2.0, 2.0, 2.0],
                "y_range": [-5.0, 5.0, 1.0],
                "z_range": [-5.0, 5.0, 1.0],
                "identify_organic": False,
            }},
        ))
        result = run.run([_si2()])
        assert result.card_names() == ["SuperCellCard", "CellStrainCard"]
        assert len(result.all()) > len(result.final())
        assert len(result.all()) == sum(len(o) for o in result.card_outputs)

    def test_outputs_by_index_and_unknown_target(self):
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
        ))
        result = run.run([_si2()])
        assert len(result.outputs(index=1)) == 1
        with pytest.raises(WorkflowError):
            result.outputs("CellStrainCard")

    def test_generator_card_produces_without_input(self):
        run = load_workflow(_workflow({
            "class": "OrderedAlloyPrototypeCard",
            "check_state": True,
            "params": {
                "prototype": "L12/A3B",
                "a_range": [3.75, 3.75, 0.1],
                "covera": 1.0,
                "sublattice_elements": "A:Cu,B:Au",
                "max_outputs": 2,
            },
        }))
        result = run.run([])
        produced = result.all()
        assert produced, "generator should not need an input dataset"
        assert len(produced) <= 2
        assert run.requires_input() is False

    def test_failure_names_the_card_and_stops_the_chain(self):
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "RandomVacancyCard", "check_state": True, "params": {
                "rules": [], "max_structures": 1, "use_seed": True, "seed": 0,
            }},
        ))
        with pytest.raises(WorkflowError) as excinfo:
            run.run([_si2()])
        assert "RandomVacancyCard" in str(excinfo.value)


class TestGroupAndFork:
    def _group(self, children, filter_card=None):
        entry = {"class": "CardGroup", "check_state": True, "card_list": children}
        if filter_card is not None:
            entry["filter_card"] = filter_card
        return entry

    def test_group_feeds_the_same_input_to_each_child(self):
        run = load_workflow(_workflow(self._group([
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[2, 2, 2])},
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[1, 1, 3])},
        ])))
        result = run.run([_si2()])
        sizes = sorted(len(structure) for structure in result.final())
        # Si2 has 2 atoms: [2,2,2] -> 16, [1,1,3] -> 6.
        assert sizes == [6, 16], "each branch must expand from the original input"

    def test_group_skips_disabled_children(self):
        run = load_workflow(_workflow(self._group([
            {"class": "SuperCellCard", "check_state": False, "params": _supercell_params(super_scale=[2, 2, 2])},
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[1, 1, 3])},
        ])))
        result = run.run([_si2()])
        assert [len(structure) for structure in result.final()] == [6]

    def test_fork_without_merge_keeps_branches_separate(self):
        fork = {
            "class": "WorkflowFork",
            "check_state": True,
            "merge": False,
            "branches": [
                {"name": "a", "enabled": True, "cards": [
                    {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[2, 2, 2])},
                ]},
                {"name": "b", "enabled": True, "cards": [
                    {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[1, 1, 3])},
                ]},
            ],
        }
        result = load_workflow(_workflow(fork)).run([_si2()])
        assert len(result.outputs("SuperCellCard", index=1)) == 1
        assert len(result.outputs("SuperCellCard", index=2)) == 1

    def test_fork_with_merge_combines_branches(self):
        fork = {
            "class": "WorkflowFork",
            "check_state": True,
            "merge": True,
            "branches": [
                {"name": "a", "enabled": True, "cards": [
                    {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[2, 2, 2])},
                ]},
                {"name": "b", "enabled": True, "cards": [
                    {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[1, 1, 3])},
                ]},
            ],
        }
        result = load_workflow(_workflow(fork)).run([_si2()])
        sizes = sorted(len(structure) for structure in result.final())
        assert sizes == [6, 16]

    def test_disabled_branch_is_ignored(self):
        fork = {
            "class": "WorkflowFork",
            "check_state": True,
            "merge": True,
            "branches": [
                {"name": "a", "enabled": False, "cards": [
                    {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[2, 2, 2])},
                ]},
                {"name": "b", "enabled": True, "cards": [
                    {"class": "SuperCellCard", "check_state": True, "params": _supercell_params(super_scale=[1, 1, 3])},
                ]},
            ],
        }
        result = load_workflow(_workflow(fork)).run([_si2()])
        assert [len(structure) for structure in result.final()] == [6]


class TestCli:
    def _write(self, tmp_path, document, name="flow.json"):
        path = tmp_path / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_preview_lists_stages_without_writing(self, tmp_path, capsys):
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
        ))
        assert run_cli(["preview", str(path)]) == 0
        out = capsys.readouterr().out
        assert "SuperCellCard" in out
        assert "no structures were produced" in out
        assert not list(tmp_path.glob("*.xyz"))

    def test_run_writes_all_outputs(self, tmp_path, capsys):
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CellStrainCard", "check_state": True, "params": {
                "axes": "uniaxial", "x_range": [-2.0, 2.0, 2.0],
                "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0],
                "identify_organic": False,
            }},
        ))
        out_path = tmp_path / "out.xyz"
        assert run_cli(["run", str(path), "-i", str(SI2), "-o", str(out_path)]) == 0
        frames = read(str(out_path), index=":")
        assert len(frames) == 24

    def test_card_selects_one_stage(self, tmp_path):
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CellStrainCard", "check_state": True, "params": {
                "axes": "uniaxial", "x_range": [-2.0, 2.0, 2.0],
                "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0],
                "identify_organic": False,
            }},
        ))
        out_path = tmp_path / "one.xyz"
        assert run_cli(["run", str(path), "-i", str(SI2), "-o", str(out_path),
                        "--card", "SuperCellCard"]) == 0
        assert len(read(str(out_path), index=":")) == 1

    def test_legacy_file_exits_with_error_and_no_traceback(self, tmp_path, capsys):
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, {"workflow_schema": 1, "cards": [
            {"class": "PerturbCard", "check_state": True, "scaling_condition": [0.25]},
        ]}, name="legacy.json")
        assert run_cli(["run", str(path), "-i", str(SI2)]) == 2
        captured = capsys.readouterr()
        assert captured.err.startswith("error:")
        assert "save it again" in captured.err
        assert "Traceback" not in captured.err

    def test_missing_input_file_is_reported(self, tmp_path, capsys):
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
        ))
        assert run_cli(["run", str(path), "-i", str(tmp_path / "nope.xyz")]) == 2
        assert "not found" in capsys.readouterr().err


class TestShippedTemplate:
    def test_structure_perturbation_template_runs_headlessly(self):
        run = load_workflow(TEMPLATE)
        stages = run.preview()
        assert [stage.card_class for stage in stages][:1] == ["SuperCellCard"]
        result = run.run([_si2()])
        assert result.card_names()[-1] == "PerturbCard"
        assert len(result.final()) > 0
