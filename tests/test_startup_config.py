from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_unusable_config_directory_stops_startup_with_actionable_error(
    tmp_path: Path,
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".config").write_text("not a directory", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(fake_home),
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", "import NepTrainKit.main"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    expected_path = fake_home / ".config" / "NepTrainKit"
    assert result.returncode == 2
    assert "Configuration storage is unavailable" in result.stderr
    assert str(expected_path) in result.stderr
    assert "No temporary configuration was used" in result.stderr
    assert "NotADirectoryError" in result.stderr
    assert "Traceback" not in result.stderr
