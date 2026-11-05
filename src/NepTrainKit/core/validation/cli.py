"""``nepkit-validate``: the validation library from a shell or an HPC job.

Examples::

    nepkit-validate capture train_dir/nep.txt --name v14 --star
    nepkit-validate import eos_bcc.xyz --name "Fe bcc EOS" --source DFT --note "PBE, 520 eV"
    nepkit-validate suite create FeNi <set-id> <set-id>
    nepkit-validate run train_dir/nep.txt --suite FeNi
    nepkit-validate library --gc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .library import Library
from .service import ValidationService


def _service(args) -> ValidationService:
    return ValidationService(Library(args.library), backend=args.backend)


def _suite_id(service: ValidationService, name_or_id: str) -> int:
    for suite in service.suites():
        if suite["name"] == name_or_id or str(suite["id"]) == name_or_id:
            return suite["id"]
    raise SystemExit(f"Unknown suite: {name_or_id}")


def _model_id(service: ValidationService, value: str) -> int:
    if Path(value).is_file():
        return service.capture_model(value, auto=False)["id"]
    if value.isdigit():
        return int(value)
    raise SystemExit(f"Not a model file or model id: {value}")


def _format(value) -> str:
    return "—" if value is None else f"{value:.3g}"


def _print_scorecard(card: dict) -> None:
    columns = card["columns"]
    header = ["model"] + [f"{c['set_name']}:{c['label']} [{c['unit']}]" for c in columns]
    print("\t".join(header))
    marks = {"better": "↑", "worse": "✗", "same": "", None: ""}
    for row in card["rows"]:
        name = row["name"] + (" (baseline)" if row["is_baseline"] else "")
        cells = [_format(cell["value"]) + marks[cell.get("verdict")] for cell in row["cells"]]
        print("\t".join([name] + cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nepkit-validate", description="NepTrainKit model validation library")
    parser.add_argument("--library", help="library folder (default: NEPTRAINKIT_LIBRARY or the user setting)")
    parser.add_argument("--backend", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="snapshot a nep.txt as a model version")
    capture.add_argument("model")
    capture.add_argument("--name", default="")
    capture.add_argument("--dataset")
    capture.add_argument("--star", action="store_true")

    imported = commands.add_parser("import", help="seal a labelled EXTXYZ as a benchmark set")
    imported.add_argument("file")
    imported.add_argument("--name", required=True)
    imported.add_argument("--source", required=True, choices=("DFT", "Other", "Demo"))
    imported.add_argument("--protocol", help="force a protocol instead of auto-detecting")
    imported.add_argument("--note", default="", help="physical / calculation conditions")
    imported.add_argument("--allow-large", action="store_true")

    suite = commands.add_parser("suite", help="manage benchmark suites")
    suite_commands = suite.add_subparsers(dest="suite_command", required=True)
    create = suite_commands.add_parser("create")
    create.add_argument("name")
    create.add_argument("sets", nargs="*")
    add = suite_commands.add_parser("add")
    add.add_argument("suite")
    add.add_argument("sets", nargs="+")
    baseline = suite_commands.add_parser("baseline")
    baseline.add_argument("suite")
    baseline.add_argument("model")
    baseline.add_argument("--note", default="")
    suite_commands.add_parser("list")

    run = commands.add_parser("run", help="evaluate a model on a suite and print the scorecard")
    run.add_argument("model", help="nep.txt path or model id")
    run.add_argument("--suite", required=True)
    run.add_argument("--force", action="store_true", help="ignore cached results")

    listing = commands.add_parser("list", help="list models or benchmark sets")
    listing.add_argument("what", choices=("models", "sets"))

    library = commands.add_parser("library", help="show where data lives and how much space it uses")
    library.add_argument("--gc", action="store_true", help="delete stale auto-captured versions and orphan files")
    library.add_argument("--days", type=int, default=30)

    args = parser.parse_args(argv)
    service = _service(args)
    output = None

    if args.command == "capture":
        output = service.capture_model(args.model, name=args.name, dataset_path=args.dataset, auto=False)
        if args.star:
            service.star_model(output["id"])
            output = service.get_model(output["id"])
    elif args.command == "import":
        provenance = {"source_kind": args.source, "conditions": args.note}
        output = service.import_set(
            args.file, name=args.name, provenance=provenance, protocol=args.protocol, allow_large=args.allow_large
        )
    elif args.command == "suite":
        if args.suite_command == "create":
            output = service.create_suite(args.name, args.sets)
        elif args.suite_command == "add":
            suite_id = _suite_id(service, args.suite)
            service.add_to_suite(suite_id, args.sets)
            output = service.suite_sets(suite_id)
        elif args.suite_command == "baseline":
            suite_id = _suite_id(service, args.suite)
            service.set_baseline(suite_id, _model_id(service, args.model), args.note)
            output = service.get_suite(suite_id)
        else:
            output = service.suites()
    elif args.command == "run":
        model_id = _model_id(service, args.model)
        suite_id = _suite_id(service, args.suite)
        service.run_suite(model_id, suite_id, force=args.force)
        card = service.scorecard(suite_id, [model_id])
        if not args.json:
            _print_scorecard(card)
            return 1 if any(row["regressions"] for row in card["rows"]) else 0
        output = card
    elif args.command == "list":
        output = service.models() if args.what == "models" else service.sets()
    elif args.command == "library":
        output = {"root": str(service.library.root), "usage_bytes": service.library.usage()}
        if args.gc:
            output["gc"] = service.collect_garbage(older_than_days=args.days, dry_run=False)

    if output is not None:
        print(json.dumps(output, indent=None if args.json else 2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
