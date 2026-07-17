#!/usr/bin/env python
"""Compare motif-lite training coverage against an enumerated count universe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_analyze import analyze_structures, structure_signatures


def _parts(total: int, n: int):
    if n == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for rest in _parts(total - value, n - 1):
            yield (value,) + rest


def _format_counts(elements: list[str], counts: tuple[int, ...]) -> str:
    return " ".join(f"{element}{count}" for element, count in zip(elements, counts) if count) or "none"


def enumerate_signatures(elements: list[str], centers: list[str], cn_values: list[int]) -> list[str]:
    return [entry["signature"] for entry in enumerate_reference(elements, centers, cn_values)]


def enumerate_reference(elements: list[str], centers: list[str], cn_values: list[int]) -> list[dict]:
    signatures = []
    for cn in cn_values:
        for center in centers:
            for counts in _parts(cn, len(elements)):
                signatures.append(
                    {
                        "signature": f"{center} | NN: {_format_counts(elements, counts)} | cn={cn}",
                        "center": center,
                        "cn": cn,
                        "counts": dict(zip(elements, counts)),
                    }
                )
    return sorted(signatures, key=lambda item: item["signature"])


def _parse_composition(text: str, atoms_list, symbols: list[str]) -> dict[str, float]:
    if text == "auto":
        counts: Counter[str] = Counter()
        for atoms in atoms_list:
            counts.update(atoms.get_chemical_symbols())
        raw = {symbol: counts[symbol] for symbol in symbols}
    else:
        raw = {}
        for item in text.split(","):
            if not item.strip():
                continue
            symbol, value = item.split(":", 1)
            raw[symbol.strip()] = float(value)
        for symbol in symbols:
            raw.setdefault(symbol, 0.0)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Composition probabilities must sum to a positive value")
    return {symbol: raw[symbol] / total for symbol in symbols}


def _multinomial_probability(counts: dict[str, int], composition: dict[str, float]) -> float:
    cn = sum(counts.values())
    coefficient = math.factorial(cn)
    probability = 1.0
    for symbol, count in counts.items():
        coefficient /= math.factorial(count)
        probability *= composition.get(symbol, 0.0) ** count
    return coefficient * probability


def expected_probabilities(
    reference_entries: list[dict],
    composition: dict[str, float],
    cn_weights: dict[int, float] | None = None,
) -> dict[str, float]:
    raw = {}
    for entry in reference_entries:
        raw[entry["signature"]] = cn_weights.get(entry["cn"], 1.0) if cn_weights else 1.0
        raw[entry["signature"]] *= composition.get(entry["center"], 0.0) * _multinomial_probability(
            entry["counts"],
            composition,
        )
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in raw}
    return {key: value / total for key, value in raw.items()}


def _histogram(atoms_list, cutoff: float | None, mult: float) -> Counter[str]:
    counts: Counter[str] = Counter()
    for atoms in atoms_list:
        counts.update(structure_signatures(atoms, cutoff=cutoff, mult=mult))
    return counts


def _cn_from_signature(signature: str) -> int:
    return int(signature.rsplit("cn=", 1)[1])


def _auto_cn_weights(atoms_list, cutoff: float | None, mult: float, cn_values: list[int]) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for atoms in atoms_list:
        for signature in structure_signatures(atoms, cutoff=cutoff, mult=mult):
            cn = _cn_from_signature(signature)
            if cn in cn_values:
                counts[cn] += 1
    total = sum(counts.values())
    if not total:
        return {cn: 1.0 / len(cn_values) for cn in cn_values}
    return {cn: counts[cn] / total for cn in cn_values}


def _parse_cn_weights(text: str, atoms_list, cutoff: float | None, mult: float, cn_values: list[int]) -> dict[int, float]:
    if text == "uniform":
        return {cn: 1.0 / len(cn_values) for cn in cn_values}
    if text == "auto":
        return _auto_cn_weights(atoms_list, cutoff, mult, cn_values)
    raw = {}
    for item in text.split(","):
        if not item.strip():
            continue
        cn, value = item.split(":", 1)
        raw[int(cn.strip())] = float(value)
    for cn in cn_values:
        raw.setdefault(cn, 0.0)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("CN weights must sum to a positive value")
    return {cn: raw[cn] / total for cn in cn_values}


def compare_to_reference(
    atoms_list,
    reference: list[str],
    cutoff: float | None,
    mult: float,
    top: int,
    expected: dict[str, float] | None = None,
) -> dict:
    observed = _histogram(atoms_list, cutoff, mult)
    ref = set(reference)
    observed_ref = {key for key in observed if key in ref}
    outside = {key: observed[key] for key in observed if key not in ref}
    missing = sorted(ref - observed_ref)
    total_observed = sum(observed.values())

    train_rows = [
        {
            "signature": key,
            "count": count,
            "fraction": count / total_observed if total_observed else 0.0,
            "in_reference": key in ref,
            "expected_probability": expected.get(key, 0.0) if expected else None,
        }
        for key, count in observed.most_common()
    ]

    report = {
        "reference_signature_count": len(ref),
        "observed_signature_count": len(observed),
        "observed_reference_signature_count": len(observed_ref),
        "coverage": len(observed_ref) / len(ref) if ref else 0.0,
        "missing_signature_count": len(missing),
        "outside_reference_signature_count": len(outside),
        "missing_examples": missing[:top],
        "outside_reference_examples": [
            {"signature": key, "count": count}
            for key, count in sorted(outside.items(), key=lambda item: (-item[1], item[0]))[:top]
        ],
        "top_observed": train_rows[:top],
        "histogram": train_rows,
    }
    if expected is not None:
        missing_expected = sorted(
            ((key, expected.get(key, 0.0)) for key in missing),
            key=lambda item: (-item[1], item[0]),
        )
        expected_rows = []
        for key in ref:
            observed_fraction = observed[key] / total_observed if total_observed else 0.0
            expected_probability = expected.get(key, 0.0)
            expected_rows.append(
                {
                    "signature": key,
                    "count": observed[key],
                    "fraction": observed_fraction,
                    "expected_probability": expected_probability,
                    "difference": observed_fraction - expected_probability,
                    "ratio": (observed_fraction / expected_probability) if expected_probability else None,
                }
            )
        report.update(
            {
                "expected_probability_mass_covered": sum(expected.get(key, 0.0) for key in observed_ref),
                "missing_expected_probability_mass": sum(expected.get(key, 0.0) for key in missing),
                "missing_high_expected_examples": [
                    {"signature": key, "expected_probability": value}
                    for key, value in missing_expected[:top]
                ],
                "top_underrepresented": sorted(
                    [row for row in expected_rows if row["expected_probability"] > 0],
                    key=lambda row: (row["difference"], row["signature"]),
                )[:top],
                "top_overrepresented": sorted(
                    expected_rows,
                    key=lambda row: (-row["difference"], row["signature"]),
                )[:top],
            }
        )
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["signature", "count", "fraction", "in_reference", "expected_probability"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="ASE-readable training structure file")
    parser.add_argument("--elements", required=True, help="Neighbor element universe, comma separated")
    parser.add_argument("--centers", help="Center elements, comma separated; defaults to --elements")
    parser.add_argument("--cn", required=True, help="Coordination numbers, comma separated, e.g. 12 or 8,12")
    parser.add_argument("--composition", help='Element fractions, e.g. "Ni:0.33,Co:0.33,Cr:0.34", or "auto"')
    parser.add_argument("--cn-weights", default="uniform", help='CN weights: "uniform", "auto", or e.g. "4:0.2,5:0.5,6:0.3"')
    parser.add_argument("--index", default=":")
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--natural-cutoff-mult", type=float, default=1.2)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hist-csv", type=Path)
    args = parser.parse_args(argv)

    elements = sorted(element.strip() for element in args.elements.split(",") if element.strip())
    centers = sorted(center.strip() for center in (args.centers or args.elements).split(",") if center.strip())
    cn_values = [int(value.strip()) for value in args.cn.split(",") if value.strip()]
    if not elements or not centers or not cn_values:
        raise ValueError("--elements, --centers, and --cn must be non-empty")

    atoms = read(args.train, index=args.index)
    atoms_list = atoms if isinstance(atoms, list) else [atoms]
    reference_entries = enumerate_reference(elements, centers, cn_values)
    reference = [entry["signature"] for entry in reference_entries]
    composition = None
    expected = None
    if args.composition:
        composition_symbols = sorted(set(elements) | set(centers))
        composition = _parse_composition(args.composition, atoms_list, composition_symbols)
        cn_weights = _parse_cn_weights(args.cn_weights, atoms_list, args.cutoff, args.natural_cutoff_mult, cn_values)
        expected = expected_probabilities(reference_entries, composition, cn_weights)
    else:
        cn_weights = None
    report = compare_to_reference(
        atoms_list,
        reference,
        cutoff=args.cutoff,
        mult=args.natural_cutoff_mult,
        top=args.top,
        expected=expected,
    )
    report["settings"] = {
        "elements": elements,
        "centers": centers,
        "cn": cn_values,
        "cutoff": args.cutoff,
        "natural_cutoff_mult": args.natural_cutoff_mult,
        "composition": composition,
        "cn_weights": cn_weights,
    }
    report["training"] = {
        key: value
        for key, value in analyze_structures(atoms_list, cutoff=args.cutoff, mult=args.natural_cutoff_mult).items()
        if key in ["total_structures", "total_environments", "unique_signature_count", "entropy", "normalized_entropy"]
    }

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    if args.hist_csv:
        _write_csv(args.hist_csv, report["histogram"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
