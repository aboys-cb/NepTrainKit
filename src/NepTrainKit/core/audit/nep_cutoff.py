"""NEP model cutoff parsing for local-chemistry audits."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from os import PathLike
from pathlib import Path

from ase.data import atomic_numbers


@dataclass(frozen=True)
class NepCutoffProfile:
    elements: tuple[str, ...]
    radial_cutoffs: tuple[float, ...]
    angular_cutoffs: tuple[float, ...]

    def pair_cutoff(self, first: str, second: str, scope: str) -> float:
        if scope == "radial":
            cutoffs = self.radial_cutoffs
        elif scope == "angular":
            cutoffs = self.angular_cutoffs
        else:
            raise ValueError("NEP cutoff scope must be 'radial' or 'angular'.")

        try:
            first_index = self.elements.index(first)
            second_index = self.elements.index(second)
        except ValueError as exc:
            raise ValueError("An element is not declared in the NEP model header.") from exc
        return 0.5 * (cutoffs[first_index] + cutoffs[second_index])


def _parse_header(line: str) -> tuple[str, ...]:
    tokens = line.split()
    if len(tokens) < 3 or not tokens[0].startswith("nep"):
        raise ValueError("NEP model header is malformed.")
    try:
        element_count = int(tokens[1])
    except ValueError as exc:
        raise ValueError("NEP model header has an invalid element count.") from exc
    if element_count <= 0 or len(tokens) != element_count + 2:
        raise ValueError("NEP model header does not match the declared element count.")

    elements = tuple(tokens[2:])
    if len(set(elements)) != len(elements) or any(element not in atomic_numbers for element in elements):
        raise ValueError("NEP model header contains an unknown or duplicate element.")
    return elements


def _parse_cutoffs(tokens: list[str], elements: tuple[str, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = tokens[1:]
    element_count = len(elements)
    if len(values) not in {4, 2 * element_count + 2}:
        raise ValueError("NEP cutoff line does not match the declared element count.")
    try:
        numeric_values = tuple(float(value) for value in values)
    except ValueError as exc:
        raise ValueError("NEP cutoff line contains a non-numeric value.") from exc

    if len(values) == 4:
        radial = (numeric_values[0],) * element_count
        angular = (numeric_values[1],) * element_count
    else:
        radial = tuple(numeric_values[2 * index] for index in range(element_count))
        angular = tuple(numeric_values[2 * index + 1] for index in range(element_count))

    if any(not isfinite(value) or value <= 0.0 for value in radial + angular):
        raise ValueError("NEP cutoffs must be finite and positive.")
    if any(angular_value > radial_value for radial_value, angular_value in zip(radial, angular)):
        raise ValueError("A NEP angular cutoff cannot exceed its radial cutoff.")
    return radial, angular


def parse_nep_cutoff(path: str | PathLike[str]) -> NepCutoffProfile:
    """Parse the header and first cutoff line from a NEP model file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        header_index, header = next(
            (index, line.strip()) for index, line in enumerate(lines) if line.strip()
        )
    except StopIteration as exc:
        raise ValueError("NEP model file is empty.") from exc

    elements = _parse_header(header)
    cutoff_tokens = next(
        (line.split() for line in lines[header_index + 1 :] if line.split() and line.split()[0] == "cutoff"),
        None,
    )
    if cutoff_tokens is None:
        raise ValueError("NEP model does not contain a cutoff line.")
    radial, angular = _parse_cutoffs(cutoff_tokens, elements)
    return NepCutoffProfile(elements, radial, angular)
