"""Synchronise Show NEP documentation icons from the application resources."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_SOURCE_DIR = REPO_ROOT / "docs" / "source"
APP_ICON_DIR = REPO_ROOT / "src" / "NepTrainKit" / "src" / "images"
DOC_ICON_DIR = DOC_SOURCE_DIR / "_static" / "image" / "generated" / "show_nep_icons"
ICON_REF_RE = re.compile(r"(?:generated/)?show_nep_icons/([A-Za-z0-9_.-]+\.(?:svg|png))")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_icon_names() -> list[str]:
    names: set[str] = set()
    for suffix in ("*.md", "*.rst"):
        for path in DOC_SOURCE_DIR.rglob(suffix):
            text = path.read_text(encoding="utf-8")
            names.update(ICON_REF_RE.findall(text))
    return sorted(names)


def sync_icons() -> dict[str, object]:
    names = referenced_icon_names()
    if not names:
        raise RuntimeError("No Show NEP icon references found in documentation.")

    DOC_ICON_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in names:
        source = APP_ICON_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing application icon: {source}")
        target = DOC_ICON_DIR / name
        shutil.copy2(source, target)
        copied.append(
            {
                "name": name,
                "source": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
                "target": str(target.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": file_sha256(target),
            }
        )

    expected = {item["name"] for item in copied}
    for path in DOC_ICON_DIR.iterdir():
        if path.name == "icon_manifest.json":
            continue
        if path.is_file() and path.name not in expected:
            path.unlink()

    manifest = {
        "source_dir": str(APP_ICON_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "target_dir": str(DOC_ICON_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "icons": copied,
    }
    manifest_path = DOC_ICON_DIR / "icon_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = sync_icons()
    print(f"Synced {len(manifest['icons'])} Show NEP icons to {manifest['target_dir']}.")


if __name__ == "__main__":
    main()
