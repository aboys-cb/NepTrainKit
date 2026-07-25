"""Small helpers for translated card combo boxes."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QCoreApplication


def _combo_label(owner, label: str) -> str:
    """Translate combo labels that are passed through stable data keys."""
    translated = QCoreApplication.translate("CardCombo", label)
    if translated != label:
        return translated
    return owner.tr(label)


def add_translated_items(owner, combo, items: Iterable[str | tuple[str, str]]) -> None:
    """Add translated combo labels while storing stable string keys in userData."""
    for item in items:
        if isinstance(item, tuple):
            value, label = item
        else:
            value = label = item
        combo.addItem(_combo_label(owner, label), userData=value)


def _translation_catalog() -> None:
    """Literal catalog for lupdate; add translated combo labels here."""
    QCoreApplication.translate("CardCombo", "Period (L_D)")
    QCoreApplication.translate("CardCombo", "Angle gradient (deg/A)")
    QCoreApplication.translate("CardCombo", "Both")
    QCoreApplication.translate("CardCombo", "Clockwise")
    QCoreApplication.translate("CardCombo", "Counterclockwise")
    QCoreApplication.translate("CardCombo", "Continuous by position")
    QCoreApplication.translate("CardCombo", "Layer-locked")
    QCoreApplication.translate("CardCombo", "Existing initial magmoms")
    QCoreApplication.translate("CardCombo", "Map/default magnitude")
    QCoreApplication.translate("CardCombo", "Existing magmoms")
    QCoreApplication.translate("CardCombo", "Element map/default")
    QCoreApplication.translate("CardCombo", "Constant magnitude")
    QCoreApplication.translate("CardCombo", "Collinear (scalar)")
    QCoreApplication.translate("CardCombo", "Non-collinear (vector)")
    QCoreApplication.translate("CardCombo", "uniaxial")
    QCoreApplication.translate("CardCombo", "biaxial")
    QCoreApplication.translate("CardCombo", "triaxial")
    QCoreApplication.translate("CardCombo", "isotropic")
    QCoreApplication.translate("CardCombo", "k-vector")
    QCoreApplication.translate("CardCombo", "group A/B")
    QCoreApplication.translate("CardCombo", "sphere")
    QCoreApplication.translate("CardCombo", "cone")
    QCoreApplication.translate("CardCombo", "plane")
    QCoreApplication.translate("CardCombo", "axis")
    QCoreApplication.translate("CardCombo", "Flip fraction")
    QCoreApplication.translate("CardCombo", "Randomize fraction")
    QCoreApplication.translate("CardCombo", "Cone disorder")
    QCoreApplication.translate("CardCombo", "x")
    QCoreApplication.translate("CardCombo", "y")
    QCoreApplication.translate("CardCombo", "z")
    QCoreApplication.translate("CardCombo", "constant volume")
    QCoreApplication.translate("CardCombo", "scale volume")
    QCoreApplication.translate("CardCombo", "free c")
    QCoreApplication.translate("CardCombo", "fixed")
    QCoreApplication.translate("CardCombo", "density")
    QCoreApplication.translate("CardCombo", "auto")
    QCoreApplication.translate("CardCombo", "general")
    QCoreApplication.translate("CardCombo", "water")
    QCoreApplication.translate("CardCombo", "ion-water")
    QCoreApplication.translate("CardCombo", "loose")
    QCoreApplication.translate("CardCombo", "dense")
    QCoreApplication.translate("CardCombo", "Grid")
    QCoreApplication.translate("CardCombo", "Sobol")
    QCoreApplication.translate("CardCombo", "Equal+Reflow")
    QCoreApplication.translate("CardCombo", "Capacity-weighted")
    QCoreApplication.translate("CardCombo", "Equal (legacy)")
    QCoreApplication.translate("CardCombo", "fcc")
    QCoreApplication.translate("CardCombo", "bcc")
    QCoreApplication.translate("CardCombo", "hcp")
    QCoreApplication.translate("CardCombo", "fcc111")
    QCoreApplication.translate("CardCombo", "A1/fcc")
    QCoreApplication.translate("CardCombo", "A2/bcc")
    QCoreApplication.translate("CardCombo", "A3/hcp")
    QCoreApplication.translate("CardCombo", "L12/A3B")
    QCoreApplication.translate("CardCombo", "B2/AB")
    QCoreApplication.translate("CardCombo", "L10/AB")
    QCoreApplication.translate("CardCombo", "Auto (Comp tag)")
    QCoreApplication.translate("CardCombo", "Manual")
    QCoreApplication.translate("CardCombo", "Exact")
    QCoreApplication.translate("CardCombo", "Random")
    QCoreApplication.translate("CardCombo", "fraction of vector")
    QCoreApplication.translate("CardCombo", "angstrom")
    QCoreApplication.translate("CardCombo", "middle")
    QCoreApplication.translate("CardCombo", "fractional")
    QCoreApplication.translate("CardCombo", "layer index")
    QCoreApplication.translate("CardCombo", "all")
    QCoreApplication.translate("CardCombo", "elements")
    QCoreApplication.translate("CardCombo", "indices")
    QCoreApplication.translate("CardCombo", "z_range")
    QCoreApplication.translate("CardCombo", "yes")
    QCoreApplication.translate("CardCombo", "no")
    QCoreApplication.translate("CardCombo", "Global canting")
    QCoreApplication.translate("CardCombo", "Single-spin tilt")
    QCoreApplication.translate("CardCombo", "Atom-pair canting")
    QCoreApplication.translate("CardCombo", "Group-pair canting")
    QCoreApplication.translate("CardCombo", "All eligible atoms")
    QCoreApplication.translate("CardCombo", "Explicit indices")
    QCoreApplication.translate("CardCombo", "Manual indices")
    QCoreApplication.translate("CardCombo", "Auto by neighbor shell")
    QCoreApplication.translate("CardCombo", "Any")
    QCoreApplication.translate("CardCombo", "Near axis")
    QCoreApplication.translate("CardCombo", "In plane (normal)")
    QCoreApplication.translate("CardCombo", "Positive only")
    QCoreApplication.translate("CardCombo", "Negative only")
    QCoreApplication.translate("CardCombo", "Both (+/- pair)")
    QCoreApplication.translate("CardCombo", "Auto from layer count")
    QCoreApplication.translate("CardCombo", "Clockwise then counterclockwise")
    QCoreApplication.translate("CardCombo", "Counterclockwise then clockwise")
    QCoreApplication.translate("CardCombo", "Cone around reference")
    QCoreApplication.translate("CardCombo", "Full random directions")
    QCoreApplication.translate("CardCombo", "exponential")
    QCoreApplication.translate("CardCombo", "squared exponential")


def combo_value(combo, fallback: str = "") -> str:
    """Return the stable item key, falling back to visible text for legacy widgets."""
    value = combo.currentData()
    if value is None:
        return combo.currentText() or fallback
    return str(value)


def set_combo_value(combo, value: object) -> None:
    """Select a combo item by stable key, falling back to visible text."""
    text = str(value)
    index = combo.findData(text)
    if index >= 0:
        combo.setCurrentIndex(index)
    else:
        combo.setCurrentText(text)
