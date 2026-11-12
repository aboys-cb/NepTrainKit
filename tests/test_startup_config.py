from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_test_session_isolates_persistent_config_and_runtime():
    from NepTrainKit.config import Config
    from NepTrainKit.paths import get_user_config_path

    root = Path(os.environ["NEPTRAINKIT_LIBRARY"]).parent
    assert get_user_config_path() == root / "config"
    assert Path(Config._instance.engine.url.database).parent == root / "config"
    assert Path(os.environ["NEPTRAINKIT_RUNTIME_ROOT"]) == root / "runtime"


def test_headless_session_has_explicit_vispy_dpi():
    from vispy import config

    if os.environ.get("QT_QPA_PLATFORM") in {"offscreen", "minimal"}:
        assert config["dpi"] == 96


def test_unusable_config_directory_stops_startup_with_actionable_error(
    tmp_path: Path,
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env = os.environ.copy()
    if os.name == "nt":
        blocked_parent = fake_home / "LocalAppData"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        env["LOCALAPPDATA"] = str(blocked_parent)
        expected_path = blocked_parent / "NepTrainKit"
    else:
        blocked_parent = fake_home / ".config"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        env["HOME"] = str(fake_home)
        expected_path = blocked_parent / "NepTrainKit"
    env.update(
        {
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

    assert result.returncode == 2
    assert "Configuration storage is unavailable" in result.stderr
    assert str(expected_path) in result.stderr
    assert "No temporary configuration was used" in result.stderr
    assert "Details:" in result.stderr
    assert "Traceback" not in result.stderr
