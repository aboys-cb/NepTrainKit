"""Widgets for displaying trustworthy per-frame structural information."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    InfoBadge,
    InfoLevel,
    SimpleCardWidget,
    StrongBodyLabel,
    setFont,
)
from qfluentwidgets.components.widgets.card_widget import CardSeparator

from NepTrainKit.core.audit import (
    StructurePhaseEvidence,
    reference_crystallography,
)
from NepTrainKit.core.structure_inspection import StructureInspection


class StructureInfoWidget(QWidget):
    """Compact card that prioritizes one frame's phase and quality signals."""

    _ORDERED_LOCAL_PHASES = ("fcc", "hcp", "bcc", "unresolved")
    _CONFIRMED_PROTOTYPES = {
        "diamond",
        "l10",
        "l12",
        "b1",
        "b2",
        "b3",
        "b4",
        "fluorite",
        "nias",
        "d03",
        "l21",
        "c1b",
        "d019",
        "c14",
        "c15",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metric_values: dict[str, StrongBodyLabel] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.card = SimpleCardWidget(self)
        self.card.setObjectName("structureInspectorCard")
        outer.addWidget(self.card)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 8, 12, 8)
        card_layout.setSpacing(3)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = StrongBodyLabel(self.tr("Structure information"), self.card)
        setFont(self.title_label, 14, QFont.Weight.DemiBold)
        self.phase_badge = InfoBadge(self.card, InfoLevel.ATTENTION)
        self.phase_badge.setText(self.tr("Not analyzed"))
        self.phase_badge.setToolTip(
            self.tr(
                "Structure-level phase evidence combines a-CNA with ordered-phase refinement."
            )
        )
        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.phase_badge)
        card_layout.addLayout(header_layout)

        self.crystallography_label = CaptionLabel(self.card)
        self.crystallography_label.setWordWrap(True)
        self.crystallography_label.setToolTip(
            self.tr(
                "Reference values describe the matched ideal prototype; "
                "the distorted snapshot may have lower instantaneous symmetry."
            )
        )
        self.crystallography_label.hide()
        card_layout.addWidget(self.crystallography_label)

        self.phase_summary_label = CaptionLabel(self.card)
        self.phase_summary_label.setWordWrap(True)
        self.phase_summary_label.setText(self.tr("Local topology evidence has not been analyzed."))
        self.phase_summary_label.setToolTip(
            self.tr(
                "Specific prototypes use separate geometry and species-ordering checks; "
                "a-CNA only reports FCC, HCP, and BCC local environments. "
                "A face-centered cubic Bravais lattice does not by itself make every site "
                "an FCC a-CNA environment."
            )
        )
        card_layout.addWidget(self.phase_summary_label)

        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(0, 1, 0, 1)
        config_layout.setSpacing(8)
        self.config_label = CaptionLabel(self.tr("Config type"), self.card)
        self.config_text = BodyLabel("—", self.card)
        self.config_text.setWordWrap(True)
        self.config_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        config_layout.addWidget(self.config_label, 0, Qt.AlignTop)
        config_layout.addWidget(self.config_text, 1)
        card_layout.addLayout(config_layout)

        summary_grid = QGridLayout()
        summary_grid.setContentsMargins(0, 0, 0, 0)
        summary_grid.setHorizontalSpacing(8)
        summary_grid.setVerticalSpacing(1)
        self.formula_text = self._add_metric(summary_grid, 0, 0, self.tr("Formula"), "formula")
        self.atom_num_text = self._add_metric(summary_grid, 0, 1, self.tr("Atoms"), "atoms")
        self.volume_text = self._add_metric(summary_grid, 1, 0, self.tr("Cell volume"), "volume")
        self.density_text = self._add_metric(summary_grid, 1, 1, self.tr("Density"), "density")
        self.length_text = self._add_wide_metric(summary_grid, 2, self.tr("Cell"), "cell")
        summary_grid.setColumnStretch(1, 1)
        summary_grid.setColumnStretch(3, 1)
        card_layout.addLayout(summary_grid)

        card_layout.addWidget(CardSeparator(self.card))

        signal_header = QHBoxLayout()
        signal_header.setContentsMargins(0, 1, 0, 0)
        self.signal_title_label = StrongBodyLabel(self.tr("Frame signals"), self.card)
        setFont(self.signal_title_label, 13, QFont.Weight.Medium)
        self.contact_badge = InfoBadge(self.card, InfoLevel.ATTENTION)
        self.contact_badge.setText(self.tr("Not analyzed"))
        signal_header.addWidget(self.signal_title_label)
        signal_header.addStretch(1)
        signal_header.addWidget(self.contact_badge)
        card_layout.addLayout(signal_header)

        signal_grid = QGridLayout()
        signal_grid.setContentsMargins(0, 0, 0, 0)
        signal_grid.setHorizontalSpacing(8)
        signal_grid.setVerticalSpacing(1)
        self.shortest_text = self._add_wide_metric(
            signal_grid, 0, self.tr("Shortest contact"), "shortest"
        )
        self.per_atom_energy_text = self._add_metric(
            signal_grid, 1, 0, self.tr("Energy / atom"), "per_atom_energy"
        )
        self.maximum_force_text = self._add_metric(
            signal_grid, 1, 1, self.tr("Max |Fᵢ|"), "maximum_force"
        )
        self.rms_force_text = self._add_metric(
            signal_grid, 2, 0, self.tr("RMS |Fᵢ|"), "rms_force"
        )
        self.net_force_text = self._add_metric(
            signal_grid, 2, 1, self.tr("Net force |ΣF|"), "net_force"
        )
        signal_grid.setColumnStretch(1, 1)
        signal_grid.setColumnStretch(3, 1)
        card_layout.addLayout(signal_grid)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _add_metric(
        self,
        layout: QGridLayout,
        row: int,
        column_group: int,
        title: str,
        key: str,
    ) -> StrongBodyLabel:
        base_column = column_group * 2
        caption = CaptionLabel(title, self.card)
        value = StrongBodyLabel("—", self.card)
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(caption, row, base_column)
        layout.addWidget(value, row, base_column + 1)
        self._metric_values[key] = value
        return value

    def _add_wide_metric(
        self,
        layout: QGridLayout,
        row: int,
        title: str,
        key: str,
    ) -> StrongBodyLabel:
        caption = CaptionLabel(title, self.card)
        value = StrongBodyLabel("—", self.card)
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(caption, row, 0)
        layout.addWidget(value, row, 1, 1, 3)
        self._metric_values[key] = value
        return value

    @staticmethod
    def _number(value: float | None, unit: str, *, digits: int = 3) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        magnitude = abs(value)
        if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e4):
            text = f"{value:.3e}"
        else:
            text = f"{value:.{digits}f}"
        return f"{text} {unit}".strip()

    def show_structure_info(self, structure) -> None:
        """Populate immediately available metadata, then await heavy analysis."""
        self.atom_num_text.setText(f"{len(structure):,}")
        self.formula_text.setText(structure.html_formula or "—")
        self.formula_text.setTextFormat(Qt.RichText)
        tag = str(getattr(structure, "tag", "") or "").strip()
        self.config_text.setText(tag or "—")
        lengths = " / ".join(f"{float(value):.3f}" for value in structure.abc)
        angles = " / ".join(f"{float(value):.1f}°" for value in structure.angles)
        self.length_text.setText(f"{lengths} Å  ·  {angles}")
        try:
            volume = float(structure.volume)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            volume = None
        self.volume_text.setText(self._number(volume, "Å³"))
        self.set_analysis_pending()

    def set_analysis_pending(self) -> None:
        self.phase_badge.setLevel(InfoLevel.INFOAMTION)
        self.phase_badge.setText(self.tr("Analyzing…"))
        self.phase_summary_label.setText(self.tr("Classifying local topology for this frame…"))
        self.crystallography_label.hide()
        self.contact_badge.setLevel(InfoLevel.INFOAMTION)
        self.contact_badge.setText(self.tr("Analyzing…"))
        for key in (
            "density",
            "shortest",
            "per_atom_energy",
            "maximum_force",
            "rms_force",
            "net_force",
        ):
            self._metric_values[key].setText("—")

    def show_analysis(
        self,
        inspection: StructureInspection,
        phase: StructurePhaseEvidence | None,
    ) -> None:
        """Render cached or freshly computed phase and frame-quality evidence."""
        self.density_text.setText(self._number(inspection.mass_density, "g/cm³"))
        self.per_atom_energy_text.setText(
            self._number(inspection.per_atom_energy, "eV/atom", digits=4)
        )
        self.per_atom_energy_text.setToolTip(
            self.tr("Total energy: {value}").format(
                value=self._number(inspection.energy, "eV", digits=4)
            )
        )
        self.maximum_force_text.setText(
            self._number(inspection.maximum_force, "eV/Å")
        )
        self.rms_force_text.setText(self._number(inspection.rms_force, "eV/Å"))
        self.net_force_text.setText(self._number(inspection.net_force, "eV/Å"))

        if inspection.shortest_distance is None or inspection.shortest_pair is None:
            self.shortest_text.setText("—")
            self.contact_badge.setLevel(InfoLevel.ATTENTION)
            self.contact_badge.setText(self.tr("Unavailable"))
        else:
            pair = "–".join(inspection.shortest_pair)
            self.shortest_text.setText(
                f"{inspection.shortest_distance:.3f} Å  ·  {pair}"
            )
            if inspection.short_contacts:
                self.contact_badge.setLevel(InfoLevel.WARNING)
                self.contact_badge.setText(self.tr("Below threshold"))
                details = " · ".join(
                    f"{'–'.join(elements)} {distance:.3f} Å"
                    for elements, distance in inspection.short_contacts[:4]
                )
                self.contact_badge.setToolTip(details)
            else:
                self.contact_badge.setLevel(InfoLevel.SUCCESS)
                self.contact_badge.setText(self.tr("Within threshold"))
                self.contact_badge.setToolTip(
                    self.tr("No element-pair minimum is below the configured radius threshold.")
                )

        self.show_phase_evidence(phase)

    def show_analysis_unavailable(self) -> None:
        self.phase_badge.setLevel(InfoLevel.ATTENTION)
        self.phase_badge.setText(self.tr("Unavailable"))
        self.phase_summary_label.setText(
            self.tr("Local topology evidence is unavailable for this frame.")
        )
        self.crystallography_label.hide()
        self.contact_badge.setLevel(InfoLevel.ATTENTION)
        self.contact_badge.setText(self.tr("Unavailable"))

    def show_phase_evidence(self, phase: StructurePhaseEvidence | None) -> None:
        if phase is None:
            self.phase_badge.setLevel(InfoLevel.ATTENTION)
            self.phase_badge.setText(self.tr("Unavailable"))
            self.phase_summary_label.setText(
                self.tr("Local topology evidence is unavailable for this frame.")
            )
            self.crystallography_label.hide()
            return

        label = self._phase_display_name(phase.phase_label)
        if phase.confidence_state == "strong":
            self.phase_badge.setLevel(InfoLevel.SUCCESS)
            suffix = (
                self.tr("Confirmed prototype")
                if phase.phase_label in self._CONFIRMED_PROTOTYPES
                else self.tr("Strong evidence")
            )
            self.phase_badge.setText(f"{label} · {suffix}")
        elif phase.confidence_state == "mixed":
            self.phase_badge.setLevel(InfoLevel.WARNING)
            self.phase_badge.setText(self.tr("Mixed local structure"))
        else:
            self.phase_badge.setLevel(InfoLevel.ATTENTION)
            self.phase_badge.setText(self.tr("Unresolved"))

        self._show_reference_crystallography(phase)

        fractions = dict(phase.local_phase_fractions)
        visible = [
            f"{self._local_phase_display_name(key)} {fractions.get(key, 0.0):.1%}"
            for key in self._ORDERED_LOCAL_PHASES
            if fractions.get(key, 0.0) >= 0.001
        ]
        prefix = self.tr("a-CNA local environments (FCC/HCP/BCC only)")
        self.phase_summary_label.setText(
            f"{prefix}: {' · '.join(visible)}" if visible else f"{prefix}: —"
        )

    def _show_reference_crystallography(
        self,
        phase: StructurePhaseEvidence,
    ) -> None:
        reference = (
            reference_crystallography(phase.phase_label)
            if phase.confidence_state == "strong"
            else None
        )
        if reference is None:
            self.crystallography_label.hide()
            return
        self.crystallography_label.setText(
            self.tr(
                "Reference crystallography (ideal prototype): "
                "{pearson} · {space_group} (No. {number}) · {bravais}"
            ).format(
                pearson=reference.pearson,
                space_group=reference.space_group,
                number=reference.space_group_number,
                bravais=self._bravais_display_name(reference.bravais),
            )
        )
        self.crystallography_label.show()

    def _bravais_display_name(self, bravais: str) -> str:
        return {
            "Face-centered cubic Bravais lattice": self.tr(
                "Face-centered cubic Bravais lattice"
            ),
            "Body-centered cubic Bravais lattice": self.tr(
                "Body-centered cubic Bravais lattice"
            ),
            "Primitive hexagonal Bravais lattice": self.tr(
                "Primitive hexagonal Bravais lattice"
            ),
            "Primitive cubic Bravais lattice; FCC-derived ordering": self.tr(
                "Primitive cubic Bravais lattice; FCC-derived ordering"
            ),
            "Primitive tetragonal Bravais lattice; FCC-derived ordering": self.tr(
                "Primitive tetragonal Bravais lattice; FCC-derived ordering"
            ),
            "Primitive cubic Bravais lattice; BCC-derived ordering": self.tr(
                "Primitive cubic Bravais lattice; BCC-derived ordering"
            ),
            "Face-centered cubic Bravais lattice; BCC-derived ordering": self.tr(
                "Face-centered cubic Bravais lattice; BCC-derived ordering"
            ),
        }[bravais]

    def _local_phase_display_name(self, label: str) -> str:
        if label == "unresolved":
            return self.tr("Other / unresolved")
        return self._phase_display_name(label)

    def _phase_display_name(self, label: str) -> str:
        return {
            "fcc": "FCC",
            "hcp": "HCP",
            "bcc": "BCC",
            "diamond": self.tr("Diamond (A4)"),
            "l10": "L1₀",
            "l12": "L1₂",
            "b1": self.tr("B1 (rock-salt)"),
            "b2": "B2 (CsCl)",
            "b3": self.tr("B3 (zinc blende)"),
            "b4": self.tr("B4 (wurtzite)"),
            "fluorite": self.tr("C1 (fluorite)"),
            "nias": "B8₁ (NiAs)",
            "d03": "D0₃",
            "l21": self.tr("L2₁ (full-Heusler)"),
            "c1b": self.tr("C1ᵦ (half-Heusler)"),
            "d019": "D0₁₉",
            "c14": "C14 Laves",
            "c15": "C15 Laves",
            "mixed": self.tr("Mixed local structure"),
            "unresolved": self.tr("Unresolved"),
        }.get(label, label)


__all__ = ["StructureInfoWidget"]
