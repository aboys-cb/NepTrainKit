import numpy as np
from pathlib import Path
from NepTrainKit.core.io.utils import parse_loss_file


def test_parse_loss(tmp_path: Path):
    sample = """0 1.0\n1 0.8\n2 0.6\n3 0.5\n"""
    loss_file = tmp_path / "loss.out"
    loss_file.write_text(sample)

    steps, losses = parse_loss_file(loss_file)

    np.testing.assert_array_equal(steps, np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(losses, np.array([1.0, 0.8, 0.6, 0.5]))
