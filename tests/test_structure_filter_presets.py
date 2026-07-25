import json

import pytest

from NepTrainKit.core import filter_presets
from NepTrainKit.core.types import (
    FilterField,
    FilterLogic,
    StructureFilterCondition,
    StructureFilterSpec,
    TextMatchMode,
)


@pytest.fixture
def preset_config(monkeypatch):
    values = {}

    monkeypatch.setattr(
        filter_presets.Config,
        "set",
        lambda section, option, value: values.__setitem__((section, option), value),
    )
    monkeypatch.setattr(
        filter_presets.Config,
        "get",
        lambda section, option, fallback=None: values.get((section, option), fallback),
    )
    monkeypatch.setattr(
        filter_presets.Config,
        "list_options",
        lambda section: [option for current_section, option in values if current_section == section],
    )

    def delete(section, option):
        return int(values.pop((section, option), None) is not None)

    monkeypatch.setattr(filter_presets.Config, "delete", delete)
    return values


def _spec() -> StructureFilterSpec:
    return StructureFilterSpec(
        logic=FilterLogic.ANY,
        conditions=(
            StructureFilterCondition(
                condition_id="temporary-row-id",
                field=FilterField.CONFIG_TYPE,
                text_values=("surface", "bulk"),
                match_mode=TextMatchMode.CONTAINS,
                case_sensitive=True,
            ),
            StructureFilterCondition(
                condition_id="formula-row-id",
                field=FilterField.FORMULA,
                enabled=False,
                text_values=("Fe2O3",),
                match_mode=TextMatchMode.EXACT,
                case_sensitive=True,
            ),
        ),
    )


def test_preset_round_trip_keeps_conditions_but_regenerates_row_ids(preset_config):
    filter_presets.save_structure_filter_preset(" Fe-O cleanup ", _spec())
    raw = preset_config[(filter_presets.STRUCTURE_FILTER_PRESET_SECTION, "Fe-O cleanup")]
    payload = json.loads(raw)

    assert payload["version"] == 1
    assert all("condition_id" not in condition for condition in payload["conditions"])

    loaded = filter_presets.load_structure_filter_preset("Fe-O cleanup")
    assert loaded is not None
    assert loaded.logic == FilterLogic.ANY
    assert [condition.field for condition in loaded.conditions] == [
        FilterField.CONFIG_TYPE,
        FilterField.FORMULA,
    ]
    assert loaded.conditions[0].text_values == ("surface", "bulk")
    assert loaded.conditions[0].case_sensitive
    assert not loaded.conditions[1].enabled
    assert {condition.condition_id for condition in loaded.conditions}.isdisjoint(
        {"temporary-row-id", "formula-row-id"}
    )


def test_preset_store_lists_renames_and_deletes(preset_config):
    filter_presets.save_structure_filter_preset("zeta", _spec())
    filter_presets.save_structure_filter_preset("Alpha", _spec())

    assert filter_presets.list_structure_filter_preset_names() == ["Alpha", "zeta"]
    assert filter_presets.structure_filter_preset_exists("Alpha")
    assert filter_presets.rename_structure_filter_preset("Alpha", "Beta")
    assert filter_presets.load_structure_filter_preset("Alpha") is None
    assert filter_presets.load_structure_filter_preset("Beta") is not None
    assert filter_presets.delete_structure_filter_preset("Beta")
    assert not filter_presets.structure_filter_preset_exists("Beta")


@pytest.mark.parametrize("name", ["", "   ", "x" * 81])
def test_preset_store_rejects_invalid_names(preset_config, name):
    with pytest.raises(ValueError):
        filter_presets.save_structure_filter_preset(name, _spec())


def test_preset_store_rejects_empty_or_incomplete_conditions(preset_config):
    with pytest.raises(ValueError, match="enabled filter condition"):
        filter_presets.save_structure_filter_preset("empty", StructureFilterSpec())

    incomplete = StructureFilterSpec(
        conditions=(
            StructureFilterCondition(
                condition_id="blank",
                field=FilterField.CONFIG_TYPE,
                text_values=(),
                match_mode=TextMatchMode.CONTAINS,
            ),
            StructureFilterCondition(
                condition_id="valid",
                field=FilterField.CONFIG_TYPE,
                text_values=("surface",),
                match_mode=TextMatchMode.CONTAINS,
            ),
        )
    )
    with pytest.raises(ValueError, match="empty filter conditions"):
        filter_presets.save_structure_filter_preset("incomplete", incomplete)


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        '{"version":99,"logic":"all","conditions":[]}',
        '{"version":1,"logic":"all","conditions":[{"field":"unknown"}]}',
        '{"version":1,"logic":"all","conditions":['
        '{"field":"config_type","enabled":true,"text_values":"surface",'
        '"match_mode":"contains","case_sensitive":false}]}',
        '{"version":1,"logic":"all","conditions":['
        '{"field":"config_type","enabled":"false","text_values":["surface"],'
        '"match_mode":"contains","case_sensitive":false}]}',
        '{"version":1,"logic":"all","conditions":['
        '{"field":"element_required","enabled":true,"text_values":["Fe"],'
        '"match_mode":"exact","case_sensitive":false}]}',
        '{"version":1,"logic":"all","conditions":['
        '{"field":"formula","enabled":true,"text_values":[""],'
        '"match_mode":"exact","case_sensitive":true}]}',
    ],
)
def test_preset_store_fails_closed_for_corrupt_or_future_records(preset_config, raw):
    preset_config[(filter_presets.STRUCTURE_FILTER_PRESET_SECTION, "broken")] = raw

    assert filter_presets.load_structure_filter_preset("broken") is None
