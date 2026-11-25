"""Rotational Ward consistency of co-located reference labels."""

from dataclasses import dataclass
from math import hypot

import numpy as np


VIRIAL_ORDER_KEY = "neptrain_virial_order"
WARD_UNIT = "μeV/atom"


@dataclass(frozen=True)
class WardLabels:
    virial: tuple[float, float, float] | None
    magnetic: tuple[float, float, float] | None
    residual: tuple[float, float, float] | None
    issues: tuple[str, ...]

    @property
    def norm(self) -> float | None:
        return hypot(*self.residual) if self.residual is not None else None


def _finite_array(value, shape):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    return array if array.shape == shape and np.isfinite(array).all() else None


def ward_labels(structure) -> WardLabels | None:
    """Return A, P, and A−P in μeV/atom, retaining any usable partial evidence."""
    props = structure.atomic_properties
    fields = structure.additional_fields
    force_keys = [key for key in ("mforce", "force_mag") if key in props]
    if "spin" not in props and not force_keys:
        return None

    atoms = len(structure)
    if not atoms:
        return WardLabels(None, None, None, ("No atoms",))

    scale = 1e6 / (2 * atoms)
    issues = []
    virial = magnetic = residual = None
    tensor = _finite_array(fields.get("virial"), (9,))
    order = fields.get(VIRIAL_ORDER_KEY)
    if tensor is None:
        issues.append("Full nine-component virial required")
    elif not isinstance(order, str) or order not in ("row_major", "column_major"):
        issues.append("Virial order is unconfirmed")
    else:
        tensor = tensor.reshape(3, 3, order="C" if order == "row_major" else "F")
        virial = scale * (tensor[[1, 2, 0], [2, 0, 1]] - tensor[[2, 0, 1], [1, 2, 0]])

    spins = _finite_array(props.get("spin"), (atoms, 3))
    forces = [_finite_array(props[key], (atoms, 3)) for key in force_keys]
    if spins is None:
        issues.append("Valid Cartesian spin vectors required")
    if not forces or any(force is None for force in forces):
        issues.append("Valid magnetic forces required")
    elif len(forces) == 2 and not np.array_equal(forces[0], forces[1]):
        issues.append("mforce and force_mag disagree")
    elif spins is not None:
        magnetic = -scale * np.cross(spins, forces[0]).sum(axis=0)

    if virial is not None and magnetic is not None:
        residual = virial - magnetic
        if not np.isfinite(residual).all() or not np.isfinite(hypot(*residual)):
            issues.append("Non-finite Ward residual")
            residual = None
    return WardLabels(
        tuple(virial.tolist()) if virial is not None else None,
        tuple(magnetic.tolist()) if magnetic is not None else None,
        tuple(residual.tolist()) if residual is not None else None,
        tuple(issues),
    )


def select_ward_residuals(indexed_structures, threshold: float) -> tuple[list[int], int]:
    """Return source indices strictly above the threshold and evaluable count."""
    selected = []
    evaluable = 0
    for index, structure in indexed_structures:
        labels = ward_labels(structure)
        norm = labels.norm if labels is not None else None
        if norm is not None:
            evaluable += 1
            if norm > threshold:
                selected.append(int(index))
    return selected, evaluable
