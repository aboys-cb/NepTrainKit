"""Update translation catalogs and build bilingual Sphinx documentation."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "docs/source"
BUILD_DIR = REPO_ROOT / "docs/_build"
GETTEXT_DIR = BUILD_DIR / "gettext"
HTML_DIR = BUILD_DIR / "html"
LOCALE_DIR = SOURCE_DIR / "locale"
LANGUAGES = ("zh_CN", "en")
HAN_TEXT = re.compile(r"[\u3400-\u9fff]")
GENERIC_TRANSLATION_PLACEHOLDERS = (
    "Details for this option",
    "This section explains the option",
    "See the parameter description for details",
)


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
            if any(
                placeholder in message_text
                for placeholder in GENERIC_TRANSLATION_PLACEHOLDERS
            ):
                errors.append(
                    f"{relative_path}: generic placeholder translation for {message_id!r}"
                )
                continue
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


def check_catalog_freshness() -> None:
    """Fail when the English catalog has not been updated for current sources."""
    from babel.messages.pofile import read_po

    with tempfile.TemporaryDirectory(prefix="neptrainkit-gettext-") as directory:
        gettext_dir = Path(directory)
        run(
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-a",
            "-q",
            "-b",
            "gettext",
            str(SOURCE_DIR),
            str(gettext_dir),
        )
        missing: list[str] = []
        for template_path in sorted(gettext_dir.rglob("*.pot")):
            relative_path = template_path.relative_to(gettext_dir)
            catalog_path = LOCALE_DIR / "en/LC_MESSAGES" / relative_path.with_suffix(".po")
            if not catalog_path.exists():
                missing.append(f"{relative_path}: English catalog is missing")
                continue
            with template_path.open(encoding="utf-8") as template_file:
                template = read_po(template_file)
            with catalog_path.open(encoding="utf-8") as catalog_file:
                catalog = read_po(catalog_file)
            catalog_ids = {message.id for message in catalog if message.id}
            for message in template:
                message_id = message.id
                if not message_id:
                    continue
                message_text = (
                    message_id
                    if isinstance(message_id, str)
                    else " ".join(message_id)
                )
                if HAN_TEXT.search(message_text) and message_id not in catalog_ids:
                    missing.append(
                        f"{relative_path}: source message is absent from English catalog {message_text!r}"
                    )
        if missing:
            raise SystemExit(
                "English catalog freshness check failed; run "
                "`python tools/docs/i18n.py update`:\n- "
                + "\n- ".join(missing)
            )
    print("Validated English catalog freshness against current documentation sources.")


class _VisibleTextParser(HTMLParser):
    """Collect user-visible HTML text while ignoring scripts and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and HAN_TEXT.search(data):
            self.text.append(" ".join(data.split()))


def check_english_html() -> None:
    """Fail when rendered English pages contain visible Chinese text."""
    html_root = HTML_DIR / "en"
    errors: list[str] = []
    for html_path in sorted(html_root.rglob("*.html")):
        html_text = html_path.read_text(encoding="utf-8")
        if "\\[::\\]" in html_text:
            errors.append(
                f"{html_path.relative_to(html_root)}: broken display-math placeholder '::'"
            )
        if "<pre><span></span>::\n" in html_text:
            errors.append(
                f"{html_path.relative_to(html_root)}: broken literal-block placeholder '::'"
            )
        parser = _VisibleTextParser()
        parser.feed(html_text)
        for text in parser.text:
            errors.append(f"{html_path.relative_to(html_root)}: {text!r}")
    if errors:
        raise SystemExit(
            "Rendered English documentation contains Chinese text:\n- "
            + "\n- ".join(errors)
        )
    print("Validated rendered English HTML contains no Chinese text.")


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
    check_catalog_freshness()
    check_english_catalogs()
    for language in LANGUAGES:
        build_language(language)
    check_english_html()
    write_landing_page()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("update", "check", "build", "all"))
    args = parser.parse_args()
    if args.command in {"update", "all"}:
        update_catalogs()
    if args.command == "check":
        check_catalog_freshness()
        check_english_catalogs()
    if args.command in {"build", "all"}:
        build_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
