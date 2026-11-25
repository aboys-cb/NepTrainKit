from pathlib import Path
from unittest.mock import Mock

import pytest

from NepTrainKit.core.io import load_result_data
from NepTrainKit.core.io import registry as result_registry


def test_result_loader_propagates_failure_after_matching(
    monkeypatch, tmp_path: Path
):
    class FailingLoader:
        name = "failing"

        def matches(self, _path):
            return True

        def load(self, _path):
            raise RuntimeError("unsupported model format")

    monkeypatch.setattr(result_registry, "_RESULT_LOADERS", [FailingLoader()])

    with pytest.raises(RuntimeError, match="unsupported model format"):
        load_result_data(tmp_path / "train.xyz")


def test_other_loader_has_diagnostic_name():
    assert result_registry.OtherLoader.name == "other"


def test_xyz_potential_picker_runs_only_for_standard_nep_data(monkeypatch, tmp_path):
    path = tmp_path / "train.xyz"
    path.write_text("1\nProperties=species:S:1:pos:R:3\nSi 0 0 0\n")
    potential = tmp_path / "chosen.txt"
    choose = Mock(return_value={"nep_txt_path": potential})
    load = Mock()
    monkeypatch.setattr(result_registry.NepTrainResultData, "from_path", load)
    assert load_result_data(path, nep_model_options=choose) is load.return_value
    choose.assert_called_once_with(path)
    load.assert_called_once_with(str(path), model_type=0, nep_txt_path=potential)

    from NepTrainKit.core.io.tace import TaceResultData

    path.write_text("1\nTACE_energy=1\nSi 0 0 0\n")
    choose.reset_mock()
    monkeypatch.setattr(TaceResultData, "from_path", load)
    load_result_data(path, nep_model_options=choose)
    choose.assert_not_called()


def test_associated_model_keeps_official_outputs_without_weights(tmp_path):
    from NepTrainKit.core.io.nep import associated_nep_model

    path = tmp_path / "train.xyz"
    assert associated_nep_model(path) is None
    for name in ("energy", "force", "virial", "stress"):
        (tmp_path / f"{name}_train.out").touch()
    assert associated_nep_model(path) == tmp_path / "nep.txt"
