from pathlib import Path

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
