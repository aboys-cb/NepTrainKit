"""Compact, UI-independent metrics for inspecting one structure frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from NepTrainKit.core.structure import table_info


AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


@dataclass(frozen=True)
class StructureInspection:
    """Scientifically named values shown by the per-frame structure inspector."""

    volume: float | None
    mass_density: float | None
    energy: float | None
    per_atom_energy: float | None
    maximum_force: float | None
    rms_force: float | None
    net_force: float | None
    shortest_distance: float | None
    shortest_pair: tuple[str, str] | None
    short_contacts: tuple[tuple[tuple[str, str], float], ...] = ()


def _mass_density(structure, volume: float | None) -> float | None:
    if volume is None or not np.isfinite(volume) or volume <= 0.0:
        return None
    try:
        total_mass = sum(
            float(atomic_masses[atomic_numbers[str(symbol)]])
            for symbol in structure.elements
        )
    except (KeyError, TypeError, ValueError):
        return None
    return total_mass / volume * AMU_PER_A3_TO_G_PER_CM3


def _short_contact_threshold(pair: tuple[str, str], coefficient: float) -> float:
    first = table_info[str(atomic_numbers[pair[0]])]["radii"] / 100.0
    second = table_info[str(atomic_numbers[pair[1]])]["radii"] / 100.0
    return float(first + second) * coefficient


def inspect_structure(
    structure,
    *,
    radius_coefficient: float = 0.7,
) -> StructureInspection:
    """Calculate one frame's geometry, energy, and force inspection values."""
    try:
        volume_value = float(structure.volume)
        volume = volume_value if np.isfinite(volume_value) else None
    except (TypeError, ValueError, np.linalg.LinAlgError):
        volume = None

    energy = None
    per_atom_energy = None
    if bool(getattr(structure, "has_energy", False)):
        try:
            energy_value = float(structure.energy)
            if np.isfinite(energy_value):
                energy = energy_value
                if len(structure) > 0:
                    per_atom_energy = energy_value / len(structure)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    maximum_force = None
    rms_force = None
    net_force = None
    if bool(getattr(structure, "has_forces", False)):
        try:
            forces = np.asarray(structure.forces, dtype=np.float64).reshape(-1, 3)
            if forces.size and np.all(np.isfinite(forces)):
                magnitudes = np.linalg.norm(forces, axis=1)
                maximum_force = float(np.max(magnitudes))
                rms_force = float(np.sqrt(np.mean(np.square(magnitudes))))
                net_force = float(np.linalg.norm(np.sum(forces, axis=0)))
        except (TypeError, ValueError):
            pass

    distance_info: dict[tuple[str, str], float] = {}
    try:
        distance_info = {
            tuple(str(element) for element in pair): float(distance)
            for pair, distance in structure.get_mini_distance_info().items()
            if np.isfinite(distance)
        }
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        pass

    shortest_pair = None
    shortest_distance = None
    if distance_info:
        shortest_pair, shortest_distance = min(
            distance_info.items(), key=lambda item: item[1]
        )

    coefficient = max(0.0, float(radius_coefficient))
    short_contacts: list[tuple[tuple[str, str], float]] = []
    for pair, distance in distance_info.items():
        try:
            if distance < _short_contact_threshold(pair, coefficient):
                short_contacts.append((pair, distance))
        except (KeyError, TypeError, ValueError):
            continue
    short_contacts.sort(key=lambda item: item[1])

    return StructureInspection(
        volume=volume,
        mass_density=_mass_density(structure, volume),
        energy=energy,
        per_atom_energy=per_atom_energy,
        maximum_force=maximum_force,
        rms_force=rms_force,
        net_force=net_force,
        shortest_distance=shortest_distance,
        shortest_pair=shortest_pair,
        short_contacts=tuple(short_contacts),
    )


__all__ = ["StructureInspection", "inspect_structure"]
