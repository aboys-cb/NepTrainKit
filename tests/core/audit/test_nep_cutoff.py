from pathlib import Path

import pytest

from NepTrainKit.core.audit.nep_cutoff import parse_nep_cutoff


def _write_nep(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_shared_cutoff_uses_nep_header_element_order(tmp_path: Path):
    model = _write_nep(
        tmp_path / "nep.txt",
        "\nnep4 2 Fe Ni\ncutoff 6 3 8 4\nn_max 4 4\n",
    )

    profile = parse_nep_cutoff(model)

    assert profile.elements == ("Fe", "Ni")
    assert profile.pair_cutoff("Fe", "Ni", "radial") == pytest.approx(6.0)
    assert profile.pair_cutoff("Fe", "Ni", "angular") == pytest.approx(3.0)


def test_parse_typewise_cutoff_averages_each_pair_for_both_scopes(tmp_path: Path):
    model = _write_nep(
        tmp_path / "nep.txt",
        "nep4 2 Fe Ni\ncutoff 4 2 6 3 8 4\nn_max 4 4\n",
    )

    profile = parse_nep_cutoff(model)

    assert profile.radial_cutoffs == pytest.approx((4.0, 6.0))
    assert profile.angular_cutoffs == pytest.approx((2.0, 3.0))
    assert profile.pair_cutoff("Fe", "Fe", "radial") == pytest.approx(4.0)
    assert profile.pair_cutoff("Fe", "Ni", "radial") == pytest.approx(5.0)
    assert profile.pair_cutoff("Fe", "Ni", "angular") == pytest.approx(2.5)
    assert profile.pair_cutoff("Ni", "Ni", "angular") == pytest.approx(3.0)


def test_parse_cutoff_rejects_token_count_that_does_not_match_elements(tmp_path: Path):
    model = _write_nep(tmp_path / "nep.txt", "nep4 2 Fe Ni\ncutoff 6 3 4 8 4\n")

    with pytest.raises(ValueError, match="declared element count"):
        parse_nep_cutoff(model)


def test_parse_cutoff_rejects_invalid_cutoff_values(tmp_path: Path):
    nonfinite = _write_nep(tmp_path / "nonfinite.txt", "nep4 1 Fe\ncutoff nan 3 8 4\n")
    angular_too_large = _write_nep(tmp_path / "ordered.txt", "nep4 1 Fe\ncutoff 3 4 8 4\n")

    with pytest.raises(ValueError, match="finite and positive"):
        parse_nep_cutoff(nonfinite)
    with pytest.raises(ValueError, match="cannot exceed"):
        parse_nep_cutoff(angular_too_large)


def test_pair_cutoff_rejects_unknown_elements_and_scope(tmp_path: Path):
    model = _write_nep(tmp_path / "nep.txt", "nep4 2 Fe Ni\ncutoff 6 3 8 4\n")
    profile = parse_nep_cutoff(model)

    with pytest.raises(ValueError, match="not declared"):
        profile.pair_cutoff("Fe", "Cu", "radial")
    with pytest.raises(ValueError, match="scope"):
        profile.pair_cutoff("Fe", "Ni", "outer")
