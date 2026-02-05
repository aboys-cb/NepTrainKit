from pathlib import Path

from NepTrainKit.core.utils import is_spin_model


def test_is_spin_model_header(tmp_path: Path):
    p = tmp_path / "nep.txt"
    p.write_text("nep4_spin 1 Fe\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is True


def test_is_spin_model_spin_mode_line(tmp_path: Path):
    p = tmp_path / "nep.txt"
    p.write_text("nep4 1 Fe\nspin_mode 1\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is True


def test_is_spin_model_false(tmp_path: Path):
    p = tmp_path / "nep.txt"
    p.write_text("nep4 1 Fe\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is False

