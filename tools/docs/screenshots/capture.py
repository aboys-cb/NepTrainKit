"""Generate and check documentation UI screenshots.

Examples
--------
List available scenarios:

    python tools/docs/screenshots/capture.py list

Update one screenshot:

    python tools/docs/screenshots/capture.py update make_data_lattice_strain

Check all committed screenshots against freshly generated ones:

    python tools/docs/screenshots/capture.py check --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PySide6.QtGui import QImage


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from annotator import annotate
from registry import SCENARIOS, ScenarioSpec
from scenarios import RUNNERS, create_context, pump_events


MANIFEST_PATH = REPO_ROOT / "docs/source/_static/image/generated/screenshot_manifest.json"
CHECK_DIR = REPO_ROOT / ".tmp/docs-screenshots"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_sha256(path: Path) -> str:
    image = QImage(str(path))
    if image.isNull():
        raise RuntimeError(f"Cannot read image: {path}")
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    payload = bytes(image.constBits())
    return hashlib.sha256(payload).hexdigest()


def _image_diff_ratio(expected: Path, actual: Path, channel_threshold: int = 4) -> float:
    expected_image = QImage(str(expected))
    actual_image = QImage(str(actual))
    if expected_image.isNull() or actual_image.isNull():
        raise RuntimeError(f"Cannot read screenshot pair: {expected}, {actual}")
    if expected_image.size() != actual_image.size():
        return 1.0
    expected_image = expected_image.convertToFormat(QImage.Format.Format_RGBA8888)
    actual_image = actual_image.convertToFormat(QImage.Format.Format_RGBA8888)
    expected_bytes = bytes(expected_image.constBits())
    actual_bytes = bytes(actual_image.constBits())
    pixels = len(expected_bytes) // 4
    if pixels == 0:
        return 1.0
    different = 0
    for offset in range(0, pixels * 4, 4):
        if (
            abs(expected_bytes[offset] - actual_bytes[offset]) > channel_threshold
            or abs(expected_bytes[offset + 1] - actual_bytes[offset + 1]) > channel_threshold
            or abs(expected_bytes[offset + 2] - actual_bytes[offset + 2]) > channel_threshold
            or abs(expected_bytes[offset + 3] - actual_bytes[offset + 3]) > channel_threshold
        ):
            different += 1
    return different / pixels


def _select_specs(names: list[str], all_scenarios: bool) -> list[ScenarioSpec]:
    if all_scenarios:
        return list(SCENARIOS.values())
    if not names:
        raise SystemExit("Specify one or more scenario names, or pass --all.")
    missing = [name for name in names if name not in SCENARIOS]
    if missing:
        available = ", ".join(sorted(SCENARIOS))
        raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}. Available: {available}")
    return [SCENARIOS[name] for name in names]


def _save_manifest_entry(spec: ScenarioSpec, output: Path) -> dict[str, object]:
    return {
        "scenario": spec.name,
        "title": spec.title,
        "description": spec.description,
        "output": output.relative_to(REPO_ROOT).as_posix(),
        "window_size": list(spec.window_size),
        "sha256": _sha256(output),
    }


def _read_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(entries: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: entries[name] for name in sorted(entries)}
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def capture(spec: ScenarioSpec, output: Path) -> Path:
    """Run one scenario and write its annotated screenshot."""
    runner = RUNNERS.get(spec.runner)
    if runner is None:
        raise RuntimeError(f"No runner registered for scenario '{spec.runner}'")

    context = create_context(REPO_ROOT, spec.window_size)
    try:
        runner(context)
        pump_events(context.app, 80)
        capture_widget = context.capture_widget or context.window
        pixmap = capture_widget.grab()
        if pixmap.isNull():
            raise RuntimeError(f"Qt returned an empty screenshot for {spec.name}")
        pixmap = annotate(pixmap, capture_widget, spec.annotations, spec.title)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(output)):
            raise RuntimeError(f"Failed to save screenshot: {output}")
        return output
    finally:
        context.window.close()
        pump_events(context.app, 20)


def command_list(_args: argparse.Namespace) -> int:
    for name, spec in sorted(SCENARIOS.items()):
        print(f"{name:28} {spec.output.as_posix()}")
        if spec.description:
            print(f"{'':28} {spec.description}")
    return 0


def command_update(args: argparse.Namespace) -> int:
    specs = _select_specs(args.names, args.all)
    manifest = _read_manifest()
    for spec in specs:
        output = REPO_ROOT / spec.output
        capture(spec, output)
        manifest[spec.name] = _save_manifest_entry(spec, output)
        print(f"updated {spec.name}: {output.relative_to(REPO_ROOT).as_posix()}")
    _write_manifest(manifest)
    return 0


def command_check(args: argparse.Namespace) -> int:
    specs = _select_specs(args.names, args.all)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for spec in specs:
        expected = REPO_ROOT / spec.output
        actual = CHECK_DIR / spec.output.name
        capture(spec, actual)
        if not expected.exists():
            print(f"missing {spec.name}: {expected.relative_to(REPO_ROOT).as_posix()}")
            failed = True
            continue
        diff_ratio = _image_diff_ratio(expected, actual)
        if diff_ratio > args.max_diff_ratio:
            print(f"outdated {spec.name}")
            print(f"  expected: {expected.relative_to(REPO_ROOT).as_posix()}")
            print(f"  actual:   {actual.relative_to(REPO_ROOT).as_posix()}")
            print(f"  diff:     {diff_ratio:.4%} > {args.max_diff_ratio:.4%}")
            failed = True
        else:
            print(f"ok {spec.name} ({diff_ratio:.4%})")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NepTrainKit documentation screenshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available screenshot scenarios.")
    list_parser.set_defaults(func=command_list)

    for name, func, help_text in (
        ("update", command_update, "Generate and overwrite screenshot assets."),
        ("check", command_check, "Compare committed screenshots with freshly generated ones."),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("names", nargs="*", help="Scenario names.")
        subparser.add_argument("--all", action="store_true", help="Run all scenarios.")
        if name == "check":
            subparser.add_argument(
                "--max-diff-ratio",
                type=float,
                default=0.075,
                help="Maximum changed-pixel ratio tolerated for local UI rendering noise.",
            )
        subparser.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
