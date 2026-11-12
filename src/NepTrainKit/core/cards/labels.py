"""Reference labels are valid only for the physical state that was labelled."""

from functools import wraps

import numpy as np

def clear_reference_labels(atoms):
    atoms.calc = None
    for key in (
        "energy", "free_energy", "virial", "stress", "dipole", "magmom",
        "force", "forces", "force_mag", "mforce", "energies", "atomic_energy",
        "magmoms", "charges", "bec", "born_effective_charges", "spin_tangent",
    ):
        atoms.info.pop(key, None)
        atoms.arrays.pop(key, None)


def _physical_state(atoms):
    return (atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc,
            atoms.arrays.get("spin", atoms.arrays.get("initial_magmoms")),
            atoms.arrays.get("initial_charges"))


def invalidate_changed_labels(run_structure):
    """Preserve labels for no-ops; drop them after geometry/species/spin changes."""
    @wraps(run_structure)
    def run(self, structure, params):
        before = tuple(None if value is None else np.array(value, copy=True)
                       for value in _physical_state(structure))
        outputs = run_structure(self, structure, params)
        for atoms in outputs:
            same = all(
                (old is None and new is None)
                or (old is not None and new is not None and np.array_equal(old, new, equal_nan=True))
                for old, new in zip(before, _physical_state(atoms))
            )
            if not same:
                clear_reference_labels(atoms)
        return outputs
    return run
