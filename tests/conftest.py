from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

LOCALAPPDATA_PATH = Path(__file__).resolve().parent / "_localappdata"
LOCALAPPDATA_PATH.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(LOCALAPPDATA_PATH)
