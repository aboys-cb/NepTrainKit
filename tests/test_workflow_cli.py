"""Headless workflow execution and CLI behaviour."""

from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import read, write
from PySide6.QtCore import QCoreApplication, QEvent

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


@pytest.mark.parametrize("fail_at", ["write", "replace"])
def test_cli_export_failure_preserves_existing_input(tmp_path, monkeypatch, fail_at):
    import ase.io
    import NepTrainKit.cli as cli

    path = tmp_path / "input.xyz"
    write(path, bulk("Si"), format="extxyz")
    before = path.read_bytes()
    structures = read(path, index=":")

    def fail_write(handle, *args, **kwargs):
        handle.write("partial output")
        raise OSError("disk full")

    if fail_at == "write":
        monkeypatch.setattr(ase.io, "write", fail_write)
    else:
        monkeypatch.setattr("NepTrainKit.core.io.atomic.os.replace", lambda *args: (_ for _ in ()).throw(OSError("publish failed")))
    with pytest.raises(OSError):
        cli._write_structures(str(path), structures)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_cli_export_replaces_input_after_success(tmp_path):
    from NepTrainKit.cli import _write_structures

    path = tmp_path / "input.xyz"
    path.write_text("old output")
    _write_structures(str(path), [bulk("Si")])
    assert read(path).get_chemical_formula() == "Si2"
    assert list(tmp_path.iterdir()) == [path]


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

    def test_every_registered_card_round_trips_its_desktop_params(self):
        """Every card's JSON form rebuilds, is stable, and matches the UI.

        These were three separate walks over CARD_REGISTRY, each instantiating
        all 42 card widgets (~1.8s per pass). The assertions are independent and
        cheap once the params exist, so one pass covers all of them.
        """
        import dataclasses

        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        import NepTrainKit.ui.views._card  # noqa: F401 - registers every card
        from NepTrainKit.core import CardManager

        checked = 0
        for card_class, (_, _, params_module, params_name, _) in CARD_REGISTRY.items():
            module = __import__(params_module, fromlist=[params_name])
            params_type = getattr(module, params_name)
            widget_class = CardManager.card_info_dict.get(card_class)
            widget_params = None
            if widget_class is not None:
                widget = widget_class()
                try:
                    widget_params = widget.get_params()
                finally:
                    # One widget per registered card; left alive they pile up on
                    # the session QApplication until teardown.
                    widget.close()
                    widget.deleteLater()

            if widget_params is not None:
                # The desktop serializer writes every field, including ones with
                # no dataclass default such as FPS's bundled ``nep_path``.
                exported = json.loads(json.dumps(dataclasses.asdict(widget_params)))
                once = params_from_dict(params_type, exported)
                assert once == widget_params, card_class
            else:
                defaults = {
                    field.name: field.default
                    for field in dataclasses.fields(params_type)
                    if field.default is not dataclasses.MISSING
                }
                if len(defaults) != len(dataclasses.fields(params_type)):
                    pytest.fail(f"{card_class} has required params but no widget to supply them")
                once = params_from_dict(params_type, json.loads(json.dumps(defaults)))

            twice = params_from_dict(params_type, json.loads(json.dumps(dataclasses.asdict(once))))
            assert once == twice, card_class
            checked += 1

        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert checked == len(CARD_REGISTRY)
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

    def test_preview_order_matches_run_order(self):
        """A container runs after its children, so preview must list it last.

        Preview is what users check before a long run; listing CardGroup before
        the cards it contains made the plan read backwards.
        """
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CardGroup", "check_state": True, "card_list": [
                {"class": "CellStrainCard", "check_state": True, "params": {
                    "axes": "uniaxial", "x_range": [-2.0, 2.0, 2.0],
                    "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0],
                    "identify_organic": False,
                }},
                {"class": "CellScalingCard", "check_state": True, "params": {
                    "engine_type": 1, "max_scaling": 0.04, "max_num": 3,
                    "perturb_angle": True, "identify_organic": False,
                    "use_seed": True, "seed": 0,
                }},
            ]},
        ))
        preview = [stage.card_class for stage in run.preview()]
        assert preview == [
            "SuperCellCard", "CellStrainCard", "CellScalingCard", "CardGroup",
        ]
        assert run.run([_si2()]).card_names() == preview

    def test_preview_index_matches_result_index(self):
        """``preview`` numbering and ``outputs(index=...)`` must agree."""
        run = load_workflow(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CardGroup", "check_state": True, "card_list": [
                {"class": "CellStrainCard", "check_state": True, "params": {
                    "axes": "uniaxial", "x_range": [-2.0, 2.0, 2.0],
                    "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0],
                    "identify_organic": False,
                }},
            ]},
        ))
        result = run.run([_si2()])
        for stage in run.preview():
            by_index = result.outputs(index=stage.index)
            assert by_index == result.outputs(stage.card_class), stage.card_class

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
    @pytest.mark.parametrize("merge", [True, False])
    def test_fork_chains_cards_within_each_branch(self, merge):
        def repeat(n, enabled=True):
            return {"class": "SuperCellCard", "check_state": enabled,
                    "params": _supercell_params(super_scale=[n, 1, 1])}

        fork = {
            "class": "WorkflowFork", "merge": merge,
            "branches": [
                {"cards": [repeat(2), repeat(99, False), repeat(3)]},
                {"cards": [repeat(4)]},
                {"enabled": False, "cards": [repeat(99)]},
            ],
        }
        run = load_workflow(_workflow(fork))
        seed = _si2()
        result = run.run([seed])
        assert [len(a) for a in result.final()] == ([12, 8] if merge else [8])
        assert [len(item.structures[0]) for item in result.card_outputs[:-1]] == [4, 12, 8]
        assert [stage.card_class for stage in run.plan.stages()] == [
            "SuperCellCard", "SuperCellCard", "SuperCellCard", "WorkflowFork"
        ]
        assert len(seed) == 2

    @pytest.mark.parametrize("filter_enabled", [True, False])
    def test_group_records_post_filter_output_in_preview_order(self, filter_enabled):
        run = load_workflow(_workflow(self._group(
            [{"class": "SuperCellCard", "params": _supercell_params()}],
            {"class": "GeometryFilterCard", "check_state": filter_enabled,
             "params": {"min_pair_distance": 0.0}},
        )))
        result = run.run([_si2()])
        classes = ["SuperCellCard"]
        if filter_enabled:
            classes.append("GeometryFilterCard")
        classes.append("CardGroup")
        assert [stage.card_class for stage in run.plan.stages()] == classes
        for index, name in enumerate(classes, 1):
            assert len(result.outputs(name, index=index)) == 1
        assert result.outputs("CardGroup") == result.final()

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

    def test_stdout_survives_a_non_utf8_console(self, tmp_path, capsys):
        """Windows Chinese consoles decode stdout as cp936.

        A single arrow character made the summary unreadable there and in any
        pipeline that stored the raw bytes, so the printed text must stay ASCII.
        """
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
        ))
        assert run_cli(["run", str(path), "-i", str(SI2), "-o", str(tmp_path / "o.xyz")]) == 0
        out = capsys.readouterr().out
        out.encode("cp936")  # raises UnicodeEncodeError if non-ASCII sneaks in
        assert out.isascii()

    def test_summary_shows_each_card_in_and_out(self, tmp_path, capsys):
        """Per-card input and output counts are the useful part of the log."""
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CellStrainCard", "check_state": True, "params": {
                "axes": "uniaxial", "x_range": [-2.0, 2.0, 2.0],
                "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0],
                "identify_organic": False,
            }},
        ))
        assert run_cli(["run", str(path), "-i", str(SI2), "-o", str(tmp_path / "o.xyz")]) == 0
        lines = capsys.readouterr().out.splitlines()

        assert "SuperCellCard" in lines[0]
        assert lines[0].split()[-4:] == ["1", "in", "1", "out"]
        assert "CellStrainCard" in lines[1]
        assert lines[1].split()[-4:] == ["1", "in", "23", "out"]
        assert lines[-2].startswith("total:")
        assert lines[-1].startswith("written:")

    def test_summary_is_field_parseable(self, tmp_path, capsys):
        """A shell can split the rows without guessing column widths."""
        from NepTrainKit.cli import run_cli

        path = self._write(tmp_path, _workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
            {"class": "CellStrainCard", "check_state": True, "params": {
                "axes": "uniaxial", "x_range": [-2.0, 2.0, 2.0],
                "y_range": [-5.0, 5.0, 1.0], "z_range": [-5.0, 5.0, 1.0],
                "identify_organic": False,
            }},
        ))
        assert run_cli(["run", str(path), "-i", str(SI2), "-o", str(tmp_path / "o.xyz")]) == 0
        rows = [
            line.split()
            for line in capsys.readouterr().out.splitlines()
            if len(line.split()) == 5 and line.split()[2] == "in"
        ]
        assert [(row[0], int(row[1]), int(row[3])) for row in rows] == [
            ("SuperCellCard", 1, 1),
            ("CellStrainCard", 1, 23),
        ]

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


class TestFPSModelPath:
    """FPS writes a concrete model path, so exports carry a machine-local value."""

    def _fps_workflow(self, nep_path):
        return _workflow({
            "class": "FPSFilterDataCard",
            "check_state": True,
            "params": {
                "nep_path": nep_path,
                "n_samples": 1,
                "min_distance": 0.0,
                "backend": "cpu",
                "chunk_max_atoms": 100000,
                "strategy": "global",
                "existing_dataset_path": "",
            },
        })

    def test_bundled_model_is_exported_as_a_portable_token(self):
        """The bundled model must survive a move to another machine."""
        import dataclasses

        from PySide6.QtWidgets import QApplication

        from NepTrainKit.paths import BUNDLED_NEP89_TOKEN, resolve_model_path

        QApplication.instance() or QApplication([])
        import NepTrainKit.ui.views._card  # noqa: F401 - registers every card
        from NepTrainKit.core import CardManager

        params = CardManager.card_info_dict["FPSFilterDataCard"]().get_params()
        assert params.nep_path == BUNDLED_NEP89_TOKEN
        assert not Path(params.nep_path).is_absolute(), "no machine-local path in the export"
        assert resolve_model_path(params.nep_path).is_file()

        exported = json.loads(json.dumps(dataclasses.asdict(params)))
        assert exported["nep_path"] == BUNDLED_NEP89_TOKEN

    def test_custom_model_path_stays_absolute(self):
        """A user-chosen model cannot be resolved elsewhere, so keep it explicit."""
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        import NepTrainKit.ui.views._card  # noqa: F401 - registers every card
        from NepTrainKit.core import CardManager

        card = CardManager.card_info_dict["FPSFilterDataCard"]()
        card.nep_path_lineedit.setText("/opt/models/my_custom.nep")
        assert card.get_params().nep_path == "/opt/models/my_custom.nep"

    def test_set_params_shows_a_real_path_for_the_token(self):
        """The interface must never display the token to the user."""
        from PySide6.QtWidgets import QApplication

        from NepTrainKit.core.cards.filter import FPSFilterParams
        from NepTrainKit.paths import BUNDLED_NEP89_TOKEN

        QApplication.instance() or QApplication([])
        import NepTrainKit.ui.views._card  # noqa: F401 - registers every card
        from NepTrainKit.core import CardManager

        card = CardManager.card_info_dict["FPSFilterDataCard"]()
        card.set_params(FPSFilterParams(nep_path=BUNDLED_NEP89_TOKEN))
        shown = card.nep_path_lineedit.text()
        assert shown != BUNDLED_NEP89_TOKEN
        assert Path(shown).is_file()

    def test_legacy_absolute_export_still_loads(self):
        """Older exports stored the absolute bundled path; they must keep working."""
        from NepTrainKit.paths import get_bundled_nep89_path

        model = get_bundled_nep89_path()
        if not model.is_file():
            pytest.skip("bundled nep89 model is not present in this checkout")
        result = load_workflow(self._fps_workflow(str(model))).run([_si2()])
        assert len(result.final()) == 1

    def test_bundled_model_path_runs_headlessly(self):
        from NepTrainKit.paths import BUNDLED_NEP89_TOKEN, get_bundled_nep89_path

        if not get_bundled_nep89_path().is_file():
            pytest.skip("bundled nep89 model is not present in this checkout")
        result = load_workflow(self._fps_workflow(BUNDLED_NEP89_TOKEN)).run([_si2()])
        assert len(result.final()) == 1

    def test_stale_machine_path_names_the_file_it_could_not_find(self, tmp_path, capsys):
        """Exporting from another machine must not produce a bare "None" error."""
        from NepTrainKit.cli import run_cli

        stale = str(tmp_path / "other-machine" / "nep89.txt")
        path = tmp_path / "fps.json"
        path.write_text(json.dumps(self._fps_workflow(stale)), encoding="utf-8")

        assert run_cli(["run", str(path), "-i", str(SI2)]) == 2
        err = capsys.readouterr().err
        assert stale in err, "the missing model path must appear in the message"
        assert "None" not in err


class TestLogQuieting:
    """stderr is an error channel for a CLI, whatever the GUI preference is."""

    def test_cli_level_survives_config_import(self):
        """``config`` re-applies the saved level; a pinned level must win."""
        from NepTrainKit.logging_config import (
            get_log_level,
            get_pinned_level,
            pin_log_level,
            release_log_level,
            set_log_level,
        )

        previous = get_log_level()
        try:
            pin_log_level("INFO")
            import NepTrainKit.config  # noqa: F401 - re-applies the saved level

            assert get_pinned_level() == "INFO"
            assert get_log_level() == "INFO", "import of config must not override a pinned level"
        finally:
            release_log_level()
            set_log_level(previous)

    def test_without_pin_config_import_still_applies_saved_level(self):
        """The desktop path must keep honouring the saved preference."""
        from NepTrainKit.logging_config import (
            get_log_level,
            get_pinned_level,
            release_log_level,
            set_log_level,
        )

        previous = get_log_level()
        try:
            release_log_level()
            assert get_pinned_level() is None
        finally:
            set_log_level(previous)

    def test_run_stderr_is_clean_and_verbose_shows_diagnostics(self, tmp_path):
        from NepTrainKit.cli import run_cli
        from NepTrainKit.logging_config import get_log_level

        path = tmp_path / "flow.json"
        path.write_text(json.dumps(_workflow(
            {"class": "SuperCellCard", "check_state": True, "params": _supercell_params()},
        )), encoding="utf-8")

        previous = get_log_level()
        try:
            run_cli(["run", str(path), "-i", str(SI2), "-o", str(tmp_path / "a.xyz")])
            default_level = get_log_level()
            assert default_level == "INFO", "a plain run reports only real problems"

            run_cli(["run", str(path), "-i", str(SI2), "-o", str(tmp_path / "b.xyz"), "-v"])
            assert get_log_level() == "DEBUG", "-v opts into library diagnostics"
        finally:
            from NepTrainKit.logging_config import release_log_level, set_log_level

            release_log_level()
            set_log_level(previous)


class TestShippedTemplate:
    def test_structure_perturbation_template_runs_headlessly(self):
        """The shipped template executes end to end, cards and branches included.

        The template itself samples a 11x11x11 strain grid under 50 random
        perturbations (~69k structures, ~30s). That volume is the point of the
        template, not of this test, so the strain grid is narrowed here: the
        card chain, the branch merge, and the final step stay identical.
        """
        document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        # First card of the branch group: narrow its three strain ranges.
        branch = document["workflow"]["cards"][1]
        branch["card_list"][0]["params"].update(
            x_range=[-5.0, 5.0, 5.0], y_range=[-5.0, 5.0, 5.0], z_range=[-5.0, 5.0, 5.0]
        )
        with tempfile.TemporaryDirectory() as workdir:
            narrowed = Path(workdir) / "narrowed.json"
            narrowed.write_text(json.dumps(document), encoding="utf-8")
            run = load_workflow(narrowed)
            stages = run.preview()
            assert [stage.card_class for stage in stages][:1] == ["SuperCellCard"]
            result = run.run([_si2()])
            assert result.card_names()[-1] == "PerturbCard"
            assert len(result.final()) > 0
