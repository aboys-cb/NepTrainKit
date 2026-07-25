"""Update translation catalogs and build bilingual Sphinx documentation."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "docs/source"
BUILD_DIR = REPO_ROOT / "docs/_build"
GETTEXT_DIR = BUILD_DIR / "gettext"
HTML_DIR = BUILD_DIR / "html"
LOCALE_DIR = SOURCE_DIR / "locale"
LANGUAGES = ("zh_CN", "en")
HAN_TEXT = re.compile(r"[\u3400-\u9fff]")


def run(*args: str) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def update_catalogs() -> None:
    run(
        sys.executable,
        "-m",
        "sphinx",
        "-E",
        "-a",
        "-b",
        "gettext",
        str(SOURCE_DIR),
        str(GETTEXT_DIR),
    )
    run(
        sys.executable,
        "-m",
        "sphinx_intl",
        "update",
        "-p",
        str(GETTEXT_DIR),
        "-d",
        str(LOCALE_DIR),
        "-l",
        "en",
    )


def check_english_catalogs() -> None:
    from babel.messages.pofile import read_po

    errors: list[str] = []
    catalog_root = LOCALE_DIR / "en/LC_MESSAGES"
    catalog_paths = sorted(catalog_root.rglob("*.po"))
    if not catalog_paths:
        errors.append(f"no catalogs found under: {catalog_root}")

    for catalog_path in catalog_paths:
        relative_path = catalog_path.relative_to(catalog_root)

        with catalog_path.open(encoding="utf-8") as catalog_file:
            catalog = read_po(catalog_file)

        for message in catalog:
            message_id = message.id if isinstance(message.id, str) else " ".join(message.id)
            message_text = (
                message.string
                if isinstance(message.string, str)
                else " ".join(message.string)
            )
            if not HAN_TEXT.search(message_id):
                continue
            if not message_text.strip():
                errors.append(f"{relative_path}: untranslated message {message_id!r}")
            elif "fuzzy" in message.flags:
                errors.append(f"{relative_path}: fuzzy message {message_id!r}")
            elif HAN_TEXT.search(message_text):
                errors.append(f"{relative_path}: Chinese remains in English text {message_id!r}")

    if errors:
        raise SystemExit("English translation check failed:\n- " + "\n- ".join(errors))
    print(f"Validated all {len(catalog_paths)} English catalogs.")


def build_language(language: str) -> None:
    run(
        sys.executable,
        "-m",
        "sphinx",
        "-E",
        "-a",
        "-W",
        "--keep-going",
        "-D",
        f"language={language}",
        "-b",
        "html",
        str(SOURCE_DIR),
        str(HTML_DIR / language),
    )


def write_landing_page() -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    (HTML_DIR / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NepTrainKit Documentation</title>
  <script>
    const language = navigator.language.toLowerCase().startsWith("zh") ? "zh_CN" : "en";
    window.location.replace(`./${language}/index.html`);
  </script>
</head>
<body>
  <p><a href="./zh_CN/index.html">中文</a> · <a href="./en/index.html">English</a></p>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_all() -> None:
    check_english_catalogs()
    for language in LANGUAGES:
        build_language(language)
    write_landing_page()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("update", "check", "build", "all"))
    args = parser.parse_args()
    if args.command in {"update", "all"}:
        update_catalogs()
    if args.command == "check":
        check_english_catalogs()
    if args.command in {"build", "all"}:
        build_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
