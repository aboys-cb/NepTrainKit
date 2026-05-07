from __future__ import annotations

from ase import Atoms

from NepTrainKit.core.config_type import append_config_tag, sanitize_config_tag, stable_config_id


def test_sanitize_config_tag_strips_quotes_separators_and_whitespace():
    assert sanitize_config_tag('  "Spin | Disorder"  ') == "Spin_Disorder"


def test_append_config_tag_normalizes_deduplicates_and_skips_empty_tags():
    atoms = Atoms("H")
    atoms.info["Config_type"] = "base old"

    assert append_config_tag(atoms, " base ") == "base|old"
    assert append_config_tag(atoms, "new tag") == "base|old|newtag"
    assert append_config_tag(atoms, "new tag") == "base|old|newtag"
    assert append_config_tag(atoms, "") == "base|old|newtag"


def test_append_config_tag_can_preserve_duplicates_when_requested():
    atoms = Atoms("H")
    atoms.info["Config_type"] = "A"

    assert append_config_tag(atoms, "A", dedupe=False) == "A|A"


def test_append_config_tag_truncates_long_tag_lists_but_preserves_edges():
    atoms = Atoms("H")

    result = append_config_tag(atoms, "first", max_len=18)
    result = append_config_tag(atoms, "middle", max_len=18)
    result = append_config_tag(atoms, "last", max_len=18)

    assert len(result) <= 18
    assert result.startswith("first|")
    assert result.endswith("|last")


def test_stable_config_id_is_stable_and_depends_only_on_config_type():
    a = Atoms("H")
    b = Atoms("He")
    a.info["Config_type"] = "A|B"
    b.info["Config_type"] = "A|B"

    assert stable_config_id(a) == stable_config_id(b)
    assert 0 <= stable_config_id(a) < 1000003


def test_stable_config_id_empty_or_missing_info_returns_zero():
    assert stable_config_id(Atoms("H")) == 0
    assert stable_config_id(object()) == 0
