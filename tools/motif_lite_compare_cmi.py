#!/usr/bin/env python
"""Compare motif-lite signatures with ChemicalMotifIdentifier outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_analyze import analyze_structures, structure_signatures


def _load_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported CMI table format: {path}")


def _label_values(df, column: str | None) -> list:
    if column:
        if column not in df.columns:
            raise ValueError(f"CMI label column not found: {column}")
        return df[column].tolist()
    return list(df.index)


def _flatten_motif_lite(atoms_list, cutoff: float | None, mult: float, mode: str) -> tuple[list[str], list[str]]:
    signatures = []
    centers = []
    for atoms in atoms_list:
        signatures.extend(structure_signatures(atoms, cutoff=cutoff, mult=mult, mode=mode))
        centers.extend(atoms.get_chemical_symbols())
    return signatures, centers


def _purity(rows: dict[str, Counter]) -> float:
    total = sum(sum(row.values()) for row in rows.values())
    return sum(max(row.values()) for row in rows.values() if row) / total if total else 0.0


def _contingency(left: list[str], right: list[str]) -> dict:
    rows: dict[str, Counter] = defaultdict(Counter)
    cols: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(left, right):
        rows[a][b] += 1
        cols[b][a] += 1

    top_rows = []
    for signature, counts in sorted(rows.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        total = sum(counts.values())
        top_rows.append(
            {
                "motif_lite": signature,
                "count": total,
                "top_cmi": [
                    {"cmi": str(label), "count": count, "fraction": count / total}
                    for label, count in counts.most_common(5)
                ],
            }
        )

    return {
        "matched_environments": len(left),
        "motif_lite_unique": len(rows),
        "cmi_unique": len(cols),
        "motif_lite_to_cmi_purity": _purity(rows),
        "cmi_to_motif_lite_purity": _purity(cols),
        "top_rows": top_rows,
    }


def compare(input_path: str, cmi_path: Path, args) -> dict:
    atoms = read(input_path, index=args.index)
    atoms_list = atoms if isinstance(atoms, list) else [atoms]
    motif_report = analyze_structures(atoms_list, cutoff=args.cutoff, mult=args.natural_cutoff_mult, mode=args.mode)
    motif_signatures, centers = _flatten_motif_lite(atoms_list, args.cutoff, args.natural_cutoff_mult, args.mode)

    cmi_df = _load_table(cmi_path)
    cmi_labels = _label_values(cmi_df, args.cmi_label_column)
    cmi_counts = None
    if args.cmi_count_column:
        if args.cmi_count_column not in cmi_df.columns:
            raise ValueError(f"CMI count column not found: {args.cmi_count_column}")
        cmi_counts = Counter()
        for label, count in zip(cmi_labels, cmi_df[args.cmi_count_column]):
            cmi_counts[str(label)] += int(count)

    report = {
        "input": input_path,
        "cmi_table": str(cmi_path),
        "motif_lite": {
            key: motif_report[key]
            for key in ["total_structures", "total_environments", "unique_signature_count", "entropy", "normalized_entropy"]
        },
        "cmi": {
            "rows": len(cmi_df),
            "unique_labels": len(set(map(str, cmi_labels))),
            "label_column": args.cmi_label_column,
            "count_column": args.cmi_count_column,
        },
    }

    if cmi_counts is not None:
        total = sum(cmi_counts.values())
        report["cmi"]["histogram"] = [
            {"label": label, "count": count, "fraction": count / total if total else 0.0}
            for label, count in cmi_counts.most_common(args.top)
        ]

    if len(cmi_labels) == len(motif_signatures):
        right = [str(label) for label in cmi_labels]
        if args.prefix_center:
            right = [f"{center}|{label}" for center, label in zip(centers, right)]
        report["comparison"] = _contingency(motif_signatures, right)
    else:
        report["comparison"] = {
            "matched_environments": 0,
            "reason": (
                f"CMI rows ({len(cmi_labels)}) do not match motif-lite environments "
                f"({len(motif_signatures)}); pass full CMI df.pkl for per-atom contingency."
            ),
        }

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="ASE-readable structure file used for motif-lite")
    parser.add_argument("--cmi-table", required=True, type=Path, help="CMI df.pkl, df_microstates.pkl, csv, or parquet")
    parser.add_argument("--index", default=":", help='ASE index selector, default ":" for all frames')
    parser.add_argument("--cutoff", type=float, default=None, help="Fixed motif-lite neighbor cutoff in Angstrom")
    parser.add_argument("--natural-cutoff-mult", type=float, default=1.2, help="ASE natural cutoff multiplier")
    parser.add_argument("--mode", choices=["count", "pair"], default="count")
    parser.add_argument("--cmi-label-column", default="shell_ID", help="CMI motif label column; use empty string for index")
    parser.add_argument("--cmi-count-column", default=None, help="Optional CMI count column, e.g. count_md")
    parser.add_argument("--no-prefix-center", action="store_false", dest="prefix_center")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cmi_label_column == "":
        args.cmi_label_column = None

    report = compare(args.input, args.cmi_table, args)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
