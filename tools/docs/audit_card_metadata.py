#!/usr/bin/env python
"""Audit Make Dataset card contributor metadata.

By default this reports missing optional fields and exits successfully. Pass
``--strict`` to fail when any built-in card is missing contributor metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail on missing contributor metadata")
    args = parser.parse_args()

    from NepTrainKit.core import CardManager
    import NepTrainKit.ui.views._card  # noqa: F401 - imports built-in cards for registration

    missing_contributors = []
    missing_descriptions = []
    for class_name, metadata in sorted(CardManager.card_metadata_dict.items()):
        path = Path(metadata.source_path) if metadata.source_path else None
        if not path or path.parent.name != "_card":
            continue
        if not metadata.description:
            missing_descriptions.append(class_name)
        if not metadata.contributors:
            missing_contributors.append(class_name)

    if missing_descriptions:
        print("Cards missing description:")
        for class_name in missing_descriptions:
            print(f"  - {class_name}")
    if missing_contributors:
        print("Cards missing contributors:")
        for class_name in missing_contributors:
            print(f"  - {class_name}")

    if not missing_descriptions and not missing_contributors:
        print("All built-in cards have description and contributor metadata.")

    if args.strict and (missing_descriptions or missing_contributors):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
