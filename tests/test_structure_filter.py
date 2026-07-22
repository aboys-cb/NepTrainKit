from types import SimpleNamespace

import numpy as np
import pytest

from NepTrainKit.core.io.base import ResultData, StructureData
from NepTrainKit.core.search import StructureFilterEngine, StructureFilterValidationError
from NepTrainKit.core.types import (
    FilterField,
    FilterLogic,
    SearchType,
    StructureFilterCondition,
    StructureFilterSpec,
    TextMatchMode,
)


def _condition(field, *values, mode=None, enabled=True, condition_id="condition"):
    return StructureFilterCondition(
        condition_id=condition_id,
        field=field,
        enabled=enabled,
        text_values=tuple(values),
        match_mode=mode,
        case_sensitive=field == FilterField.FORMULA,
    )


def _result_data():
    rows = [
        SimpleNamespace(tag="surface_relax", formula="Fe2O3", elements=["Fe", "O"]),
        SimpleNamespace(tag="bulk", formula="FeO", elements=["Fe", "O"]),
        SimpleNamespace(tag="surface_h", formula="FeOH", elements=["Fe", "O", "H"]),
        SimpleNamespace(tag="molecule", formula="H2O", elements=["H", "O"]),
        SimpleNamespace(tag="dopant", formula="FeOC", elements=["Fe", "O", "C"]),
    ]
    structure = StructureData(rows)

    def search_config(expression, search_type):
        assert search_type == SearchType.EXPRESSION
        if expression == "natoms > 2":
            return [0, 2, 3, 4]
        raise ValueError("Invalid expression syntax.")

    return SimpleNamespace(structure=structure, search_config=search_config)


@pytest.mark.parametrize(
    ("mode", "value", "expected"),
    [
        (TextMatchMode.CONTAINS, "SURFACE", (0, 2)),
        (TextMatchMode.EXACT, "bulk", (1,)),
        (TextMatchMode.PREFIX, "surf", (0, 2)),
        (TextMatchMode.SUFFIX, "relax", (0,)),
        (TextMatchMode.REGEX, r"^(bulk|dopant)$", (1, 4)),
    ],
)
def test_config_type_text_modes(mode, value, expected):
    spec = StructureFilterSpec(conditions=(_condition(FilterField.CONFIG_TYPE, value, mode=mode),))
    assert StructureFilterEngine.evaluate(_result_data(), spec).indices == expected


def test_invalid_regex_is_a_validation_error():
    spec = StructureFilterSpec(
        conditions=(_condition(FilterField.CONFIG_TYPE, "[", mode=TextMatchMode.REGEX, condition_id="bad"),)
    )
    with pytest.raises(StructureFilterValidationError) as caught:
        StructureFilterEngine.evaluate(_result_data(), spec)
    assert caught.value.code == "invalid_regex"
    assert caught.value.condition_id == "bad"


def test_multiple_text_values_are_or_within_one_condition():
    data = _result_data()
    config = StructureFilterSpec(
        conditions=(
            _condition(
                FilterField.CONFIG_TYPE,
                "bulk",
                "dopant",
                mode=TextMatchMode.EXACT,
            ),
        )
    )
    formula = StructureFilterSpec(
        conditions=(
            _condition(
                FilterField.FORMULA,
                "Fe2O3",
                "H2O",
                mode=TextMatchMode.EXACT,
            ),
        )
    )

    assert StructureFilterEngine.evaluate(data, config).indices == (1, 4)
    assert StructureFilterEngine.evaluate(data, formula).indices == (0, 3)


def test_formula_defaults_to_exact_and_is_case_sensitive():
    spec = StructureFilterSpec(conditions=(_condition(FilterField.FORMULA, "Fe2O3"),))
    assert StructureFilterEngine.evaluate(_result_data(), spec).indices == (0,)
    wrong_case = StructureFilterSpec(conditions=(_condition(FilterField.FORMULA, "fe2o3"),))
    assert StructureFilterEngine.evaluate(_result_data(), wrong_case).indices == ()


def test_case_sensitivity_is_explicit_for_plain_text_and_regex():
    data = _result_data()
    sensitive_text = StructureFilterCondition(
        condition_id="text",
        field=FilterField.CONFIG_TYPE,
        text_values=("SURFACE",),
        match_mode=TextMatchMode.CONTAINS,
        case_sensitive=True,
    )
    insensitive_regex = StructureFilterCondition(
        condition_id="regex-insensitive",
        field=FilterField.CONFIG_TYPE,
        text_values=(r"^SURFACE",),
        match_mode=TextMatchMode.REGEX,
        case_sensitive=False,
    )
    sensitive_regex = StructureFilterCondition(
        condition_id="regex-sensitive",
        field=FilterField.CONFIG_TYPE,
        text_values=(r"^SURFACE",),
        match_mode=TextMatchMode.REGEX,
        case_sensitive=True,
    )

    assert StructureFilterEngine.evaluate(data, StructureFilterSpec(conditions=(sensitive_text,))).indices == ()
    assert StructureFilterEngine.evaluate(data, StructureFilterSpec(conditions=(insensitive_regex,))).indices == (0, 2)
    assert StructureFilterEngine.evaluate(data, StructureFilterSpec(conditions=(sensitive_regex,))).indices == ()


@pytest.mark.parametrize(
    ("mode", "value", "expected_insensitive"),
    [
        (TextMatchMode.CONTAINS, "SURFACE", (0, 2)),
        (TextMatchMode.EXACT, "BULK", (1,)),
        (TextMatchMode.PREFIX, "SURF", (0, 2)),
        (TextMatchMode.SUFFIX, "RELAX", (0,)),
        (TextMatchMode.REGEX, r"^SURFACE", (0, 2)),
    ],
)
def test_match_case_switch_applies_to_every_text_mode(mode, value, expected_insensitive):
    data = _result_data()
    insensitive = StructureFilterCondition(
        condition_id="insensitive",
        field=FilterField.CONFIG_TYPE,
        text_values=(value,),
        match_mode=mode,
        case_sensitive=False,
    )
    sensitive = StructureFilterCondition(
        condition_id="sensitive",
        field=FilterField.CONFIG_TYPE,
        text_values=(value,),
        match_mode=mode,
        case_sensitive=True,
    )

    assert (
        StructureFilterEngine.evaluate(data, StructureFilterSpec(conditions=(insensitive,))).indices
        == expected_insensitive
    )
    assert StructureFilterEngine.evaluate(data, StructureFilterSpec(conditions=(sensitive,))).indices == ()


def test_formula_can_explicitly_ignore_case():
    condition = StructureFilterCondition(
        condition_id="formula",
        field=FilterField.FORMULA,
        text_values=("fe2o3",),
        match_mode=TextMatchMode.EXACT,
        case_sensitive=False,
    )

    assert StructureFilterEngine.evaluate(
        _result_data(),
        StructureFilterSpec(conditions=(condition,)),
    ).indices == (0,)


def test_required_excluded_and_allowed_elements_have_distinct_semantics():
    data = _result_data()
    required = StructureFilterSpec(conditions=(_condition(FilterField.ELEMENT_REQUIRED, "fe", "O"),))
    excluded = StructureFilterSpec(conditions=(_condition(FilterField.ELEMENT_EXCLUDED, "H"),))
    allowed = StructureFilterSpec(conditions=(_condition(FilterField.ELEMENT_ALLOWED, "Fe", "O"),))

    assert StructureFilterEngine.evaluate(data, required).indices == (0, 1, 2, 4)
    assert StructureFilterEngine.evaluate(data, excluded).indices == (0, 1, 4)
    assert StructureFilterEngine.evaluate(data, allowed).indices == (0, 1)


def test_unknown_element_fails_closed():
    spec = StructureFilterSpec(conditions=(_condition(FilterField.ELEMENT_REQUIRED, "Xx", condition_id="element"),))
    with pytest.raises(StructureFilterValidationError) as caught:
        StructureFilterEngine.evaluate(_result_data(), spec)
    assert caught.value.code == "unknown_element"
    assert caught.value.condition_id == "element"


def test_composite_all_filter_intersects_conditions():
    spec = StructureFilterSpec(
        conditions=(
            _condition(FilterField.CONFIG_TYPE, "surface", mode=TextMatchMode.CONTAINS),
            _condition(FilterField.ELEMENT_REQUIRED, "Fe", "O"),
            _condition(FilterField.FORMULA, "Fe2O3"),
        )
    )
    assert StructureFilterEngine.evaluate(_result_data(), spec).indices == (0,)


def test_any_filter_unions_enabled_conditions_and_ignores_disabled_rows():
    spec = StructureFilterSpec(
        conditions=(
            _condition(FilterField.FORMULA, "H2O"),
            _condition(FilterField.CONFIG_TYPE, "bulk", mode=TextMatchMode.EXACT),
            _condition(FilterField.CONFIG_TYPE, "surface", enabled=False),
        ),
        logic=FilterLogic.ANY,
    )
    assert StructureFilterEngine.evaluate(_result_data(), spec).indices == (1, 3)


def test_expression_uses_existing_expression_engine():
    spec = StructureFilterSpec(conditions=(_condition(FilterField.EXPRESSION, "natoms > 2"),))
    assert StructureFilterEngine.evaluate(_result_data(), spec).indices == (0, 2, 3, 4)


def test_expression_errors_are_structured():
    spec = StructureFilterSpec(
        conditions=(_condition(FilterField.EXPRESSION, "not valid", condition_id="expression"),)
    )
    with pytest.raises(StructureFilterValidationError) as caught:
        StructureFilterEngine.evaluate(_result_data(), spec)
    assert caught.value.code == "invalid_expression"
    assert caught.value.condition_id == "expression"


@pytest.mark.parametrize(
    "field",
    [
        FilterField.CONFIG_TYPE,
        FilterField.FORMULA,
        FilterField.ELEMENT_REQUIRED,
        FilterField.ELEMENT_EXCLUDED,
        FilterField.ELEMENT_ALLOWED,
        FilterField.EXPRESSION,
    ],
)
def test_blank_conditions_fail_closed_for_every_field(field):
    spec = StructureFilterSpec(conditions=(_condition(field, "  ", condition_id="blank"),))

    with pytest.raises(StructureFilterValidationError) as caught:
        StructureFilterEngine.evaluate(_result_data(), spec)

    assert caught.value.code == "empty_condition"
    assert caught.value.condition_id == "blank"


def test_all_disabled_conditions_fail_closed():
    spec = StructureFilterSpec(
        conditions=(
            _condition(FilterField.CONFIG_TYPE, "surface", enabled=False),
            _condition(FilterField.ELEMENT_REQUIRED, "Fe", enabled=False),
        )
    )

    with pytest.raises(StructureFilterValidationError) as caught:
        StructureFilterEngine.evaluate(_result_data(), spec)

    assert caught.value.code == "empty_condition"


def test_removed_structures_are_excluded_and_version_changes():
    data = _result_data()
    spec = StructureFilterSpec(conditions=(_condition(FilterField.ELEMENT_REQUIRED, "Fe"),))
    before = StructureFilterEngine.evaluate(data, spec)
    data.structure.remove(1)
    after = StructureFilterEngine.evaluate(data, spec)

    assert before.indices == (0, 1, 2, 4)
    assert after.indices == (0, 2, 4)
    assert after.dataset_version != before.dataset_version
    assert after.active_count == 4


def test_result_indices_are_sorted_unique_original_indices():
    data = _result_data()
    data.structure.remove([1, 3])
    spec = StructureFilterSpec(conditions=(_condition(FilterField.ELEMENT_REQUIRED, "O"),))
    result = StructureFilterEngine.evaluate(data, spec)
    assert result.indices == (0, 2, 4)
    assert result.indices == tuple(sorted(set(result.indices)))


def test_filter_spec_round_trip_preserves_typed_state():
    spec = StructureFilterSpec(
        conditions=(
            StructureFilterCondition(
                condition_id="formula",
                field=FilterField.FORMULA,
                enabled=False,
                text_values=("Fe2O3",),
                match_mode=TextMatchMode.EXACT,
                case_sensitive=True,
            ),
        ),
        logic=FilterLogic.ANY,
    )
    assert StructureFilterSpec.from_dict(spec.to_dict()) == spec


def test_bulk_selection_modes_record_one_undo_step(tmp_path):
    data = ResultData(tmp_path / "nep.txt", tmp_path / "data.xyz", tmp_path / "descriptor.out")
    data._atoms_dataset = _result_data().structure

    assert data.apply_selection([0, 1], "replace")
    assert data.select_index == {0, 1}
    assert data.apply_selection([2, 4], "add")
    assert data.select_index == {0, 1, 2, 4}
    assert data.apply_selection([1, 4], "remove")
    assert data.select_index == {0, 2}

    assert data.undo_selection()
    assert data.select_index == {0, 1, 2, 4}
    assert data.undo_selection()
    assert data.select_index == {0, 1}
    assert data.undo_selection()
    assert data.select_index == set()


def test_bulk_selection_rejects_stale_removed_indices(tmp_path):
    data = ResultData(tmp_path / "nep.txt", tmp_path / "data.xyz", tmp_path / "descriptor.out")
    data._atoms_dataset = _result_data().structure
    data.structure.remove(1)
    data.apply_selection([0, 1, 2], "replace")
    assert data.select_index == {0, 2}
