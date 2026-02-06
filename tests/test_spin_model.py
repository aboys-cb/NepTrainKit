from pathlib import Path
import shutil
import uuid

import pytest
from NepTrainKit.core.utils import is_spin_model


@pytest.fixture()
def local_tmp_path() -> Path:
    base_tmp = Path(__file__).resolve().parents[1] / ".tmp_localappdata" / "spin_model_tmp"
    base_tmp.mkdir(parents=True, exist_ok=True)
    tmp_path = base_tmp / f"spin_model_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    try:
        yield tmp_path
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_is_spin_model_header(local_tmp_path: Path):
    p = local_tmp_path / "nep.txt"
    p.write_text("nep4_spin 1 Fe\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is True


def test_is_spin_model_spin_mode_line(local_tmp_path: Path):
    p = local_tmp_path / "nep.txt"
    p.write_text("nep4 1 Fe\nspin_mode 1\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is True


def test_is_spin_model_false(local_tmp_path: Path):
    p = local_tmp_path / "nep.txt"
    p.write_text("nep4 1 Fe\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is False
