"""Dialogs and formatting helpers for Make Dataset card metadata."""

from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from NepTrainKit.core import CardManager, CardMetadata


def _tr(text: str) -> str:
    return QCoreApplication.translate("CardMetadata", text)


@lru_cache(maxsize=4)
def _localized_catalog(_language_marker: str):
    """Return card catalog text for the currently installed Qt translator."""
    names_and_descriptions = {
        "BainPathCard": (
            QCoreApplication.translate("CardCatalog", "Bain Path"),
            QCoreApplication.translate("CardCatalog", "Generate fixed-structure Bain/tetragonal distortion paths."),
        ),
        "CardGroup": (
            QCoreApplication.translate("CardCatalog", "Card Group"),
            QCoreApplication.translate("CardCatalog", "Group container that executes child cards in sequence and aggregates outputs."),
        ),
        "CellScalingCard": (
            QCoreApplication.translate("CardCatalog", "Lattice Perturb"),
            QCoreApplication.translate("CardCatalog", "Generate perturbed lattice structures using stochastic scaling factors."),
        ),
        "CellStrainCard": (
            QCoreApplication.translate("CardCatalog", "Lattice Strain"),
            QCoreApplication.translate("CardCatalog", "Produce strained lattice variants along user-selected axes and ranges."),
        ),
        "CompositionGradientCard": (
            QCoreApplication.translate("CardCatalog", "Composition Gradient"),
            QCoreApplication.translate("CardCatalog", "Assign atom types from a layerwise composition gradient."),
        ),
        "CompositionSweepCard": (
            QCoreApplication.translate("CardCatalog", "Composition Sweep"),
            QCoreApplication.translate("CardCatalog", "Create multiple copies per input structure, each annotated with a target composition."),
        ),
        "ConditionalReplaceCard": (
            QCoreApplication.translate("CardCatalog", "Conditional Replace"),
            QCoreApplication.translate("CardCatalog", "Replace atoms in the active structures using spatial conditions and ratios."),
        ),
        "CorrelatedRandomSpinCard": (
            QCoreApplication.translate("CardCatalog", "Correlated Random Spin"),
            QCoreApplication.translate("CardCatalog", "Generate non-collinear random spins with an explicit spatial correlation length."),
        ),
        "CrystalPrototypeBuilderCard": (
            QCoreApplication.translate("CardCatalog", "Crystal Prototype Builder"),
            QCoreApplication.translate("CardCatalog", "Generate simple bulk crystal prototypes without requiring input structures."),
        ),
        "FPSFilterDataCard": (
            QCoreApplication.translate("CardCatalog", "FPS Filter"),
            QCoreApplication.translate("CardCatalog", "Filter dataset entries via farthest point sampling computed from NEP descriptors."),
        ),
        "FoldedHelixCard": (
            QCoreApplication.translate("CardCatalog", "Folded Helix"),
            QCoreApplication.translate("CardCatalog", "Assign symmetric clockwise-then-counterclockwise layered helix moments."),
        ),
        "GeometryFilterCard": (
            QCoreApplication.translate("CardCatalog", "Geometry Filter"),
            QCoreApplication.translate("CardCatalog", "Reject structures that violate explicit geometry-quality thresholds."),
        ),
        "GroupLabelCard": (
            QCoreApplication.translate("CardCatalog", "Group Label"),
            QCoreApplication.translate("CardCatalog", "Attach atoms.arrays['group'] labels using common, lattice-agnostic rules."),
        ),
        "InsertDefectCard": (
            QCoreApplication.translate("CardCatalog", "Insert Defect"),
            QCoreApplication.translate("CardCatalog", "Create interstitial or surface-adsorbate configurations."),
        ),
        "LayerCopyCard": (
            QCoreApplication.translate("CardCatalog", "Layer Copy"),
            QCoreApplication.translate("CardCatalog", "Warp a structure by dz=f(x,y), then copy and translate it along z into one stack."),
        ),
        "LocalSolvationCard": (
            QCoreApplication.translate("CardCatalog", "Local Solvation"),
            QCoreApplication.translate("CardCatalog", "Generate local solvent shells around selected atoms."),
        ),
        "MagneticMomentRotationCard": (
            QCoreApplication.translate("CardCatalog", "Magmom Rotation"),
            QCoreApplication.translate("CardCatalog", "Rotate and optionally rescale atomic magnetic moments for selected species."),
        ),
        "MagneticOrderCard": (
            QCoreApplication.translate("CardCatalog", "Magnetic Order"),
            QCoreApplication.translate("CardCatalog", "Assign initial magnetic moments and generate common collinear spin patterns."),
        ),
        "OrganicMolConfigPBCCard": (
            QCoreApplication.translate("CardCatalog", "Organic Mol Config"),
            QCoreApplication.translate("CardCatalog", "Create torsion-driven molecular configurations using the TorsionGuard PBC workflow."),
        ),
        "PerturbCard": (
            QCoreApplication.translate("CardCatalog", "Atomic Perturb"),
            QCoreApplication.translate("CardCatalog", "Apply random atomic displacements within a configurable distance budget."),
        ),
        "RandomDopingCard": (
            QCoreApplication.translate("CardCatalog", "Random Doping"),
            QCoreApplication.translate("CardCatalog", "Perform random atomic substitutions according to user-specified doping rules."),
        ),
        "RandomOccupancyCard": (
            QCoreApplication.translate("CardCatalog", "Random Occupancy"),
            QCoreApplication.translate("CardCatalog", "Assign alloy elements to all or grouped lattice sites using a target composition."),
        ),
        "RandomPackingCard": (
            QCoreApplication.translate("CardCatalog", "Random Packing"),
            QCoreApplication.translate("CardCatalog", "Generate random atomic coordinates while preserving cell constraints."),
        ),
        "RandomSlabCard": (
            QCoreApplication.translate("CardCatalog", "Random Slab"),
            QCoreApplication.translate("CardCatalog", "Construct surface slabs across multiple Miller indices and thicknesses."),
        ),
        "RandomVacancyCard": (
            QCoreApplication.translate("CardCatalog", "Random Vacancy"),
            QCoreApplication.translate("CardCatalog", "Create vacancy structures by probabilistically removing atoms according to rules."),
        ),
        "SetMagneticMomentsCard": (
            QCoreApplication.translate("CardCatalog", "Set Magnetic Moments"),
            QCoreApplication.translate("CardCatalog", "Set or convert magnetic moments into a consistent scalar or vector representation."),
        ),
        "ShearAngleCard": (
            QCoreApplication.translate("CardCatalog", "Shear Angle Strain"),
            QCoreApplication.translate("CardCatalog", "Perturb lattice angles while preserving cell lengths."),
        ),
        "ShearMatrixCard": (
            QCoreApplication.translate("CardCatalog", "Shear Matrix Strain"),
            QCoreApplication.translate("CardCatalog", "Apply shear matrices along the principal lattice planes."),
        ),
        "SmallAngleSpinTiltCard": (
            QCoreApplication.translate("CardCatalog", "Small-Angle Spin Tilt"),
            QCoreApplication.translate("CardCatalog", "Generate deterministic single-spin small-angle tilt configurations."),
        ),
        "SolventBoxFillCard": (
            QCoreApplication.translate("CardCatalog", "Solvent Box Fill"),
            QCoreApplication.translate("CardCatalog", "Fill an existing periodic cell with solvent molecules."),
        ),
        "SpinDisorderCard": (
            QCoreApplication.translate("CardCatalog", "Spin Disorder"),
            QCoreApplication.translate("CardCatalog", "Generate spin states with explicit disorder fractions."),
        ),
        "SpinSpiralCard": (
            QCoreApplication.translate("CardCatalog", "Spin Spiral"),
            QCoreApplication.translate("CardCatalog", "Assign non-collinear spiral magnetic moments using a 1D phase field."),
        ),
        "StackingFaultCard": (
            QCoreApplication.translate("CardCatalog", "Stacking Fault"),
            QCoreApplication.translate("CardCatalog", "Generate stacking-fault or twin structures."),
        ),
        "StrictGSFEPathCard": (
            QCoreApplication.translate("CardCatalog", "Strict GSFE Path"),
            QCoreApplication.translate("CardCatalog", "Generate unrelaxed GSFE structures with an explicit plane and slip direction."),
        ),
        "SuperCellCard": (
            QCoreApplication.translate("CardCatalog", "Super Cell"),
            QCoreApplication.translate("CardCatalog", "Create supercells from fixed scale factors, target lattice lengths, or atom limits."),
        ),
        "VacancyDefectCard": (
            QCoreApplication.translate("CardCatalog", "Vacancy Defect Generation"),
            QCoreApplication.translate("CardCatalog", "Sample vacancy defects by concentration or explicit counts."),
        ),
        "VibrationModePerturbCard": (
            QCoreApplication.translate("CardCatalog", "Vib Mode Perturb"),
            QCoreApplication.translate("CardCatalog", "Generate perturbations along precomputed vibrational modes."),
        ),
    }
    groups = {
        "Alloy": QCoreApplication.translate("CardCatalog", "Alloy"),
        "Container": QCoreApplication.translate("CardCatalog", "Container"),
        "Defect": QCoreApplication.translate("CardCatalog", "Defect"),
        "Filter": QCoreApplication.translate("CardCatalog", "Filter"),
        "Lattice": QCoreApplication.translate("CardCatalog", "Lattice"),
        "Magnetism": QCoreApplication.translate("CardCatalog", "Magnetism"),
        "Organic": QCoreApplication.translate("CardCatalog", "Organic"),
        "Perturbation": QCoreApplication.translate("CardCatalog", "Perturbation"),
        "Structure": QCoreApplication.translate("CardCatalog", "Structure"),
        "Surface": QCoreApplication.translate("CardCatalog", "Surface"),
    }
    roles = {
        "author": QCoreApplication.translate("CardCatalog", "author"),
        "maintainer": QCoreApplication.translate("CardCatalog", "maintainer"),
        "contributor": QCoreApplication.translate("CardCatalog", "contributor"),
    }
    return names_and_descriptions, groups, roles


def _catalog():
    return _localized_catalog(_tr("Built-in"))


def localized_card_name(metadata: CardMetadata) -> str:
    entry = _catalog()[0].get(metadata.class_name)
    return entry[0] if entry else metadata.card_name


def localized_card_description(metadata: CardMetadata) -> str:
    entry = _catalog()[0].get(metadata.class_name)
    return entry[1] if entry else metadata.description


def localized_card_group(metadata: CardMetadata) -> str:
    return _catalog()[1].get(metadata.group, metadata.group or "")


def localized_contributor_role(role: str) -> str:
    return _catalog()[2].get(role.strip().lower(), role)


def contributor_label(contributor) -> str:
    """Return a compact public label for one contributor."""
    role = (
        f" ({localized_contributor_role(contributor.role)})"
        if contributor.role
        else ""
    )
    return f"{contributor.name}{role}"


def contributors_text(metadata: CardMetadata) -> str:
    """Return contributor names for plain-text UI surfaces."""
    if not metadata.contributors:
        return _tr("Not specified")
    return ", ".join(contributor_label(item) for item in metadata.contributors)


def card_tooltip(metadata: CardMetadata) -> str:
    """Build a short tooltip for an Add Card action."""
    lines = [localized_card_name(metadata)]
    description = localized_card_description(metadata)
    if description:
        lines.append(description)
    lines.append(_tr("Contributors: {contributors}").format(contributors=contributors_text(metadata)))
    if metadata.version:
        lines.append(_tr("Version: {version}").format(version=metadata.version))
    return "\n".join(lines)


def _source_label(metadata: CardMetadata) -> str:
    path = Path(metadata.source_path) if metadata.source_path else None
    if path and path.parent.name == "_card":
        return _tr("Built-in")
    if path:
        return _tr("Custom")
    return _tr("Unknown")


def _contributors_html(metadata: CardMetadata) -> str:
    if not metadata.contributors:
        return """
        <div class="section">
          <h3>{contributors_title}</h3>
          <p class="empty">{empty_text}</p>
        </div>
        """.format(
            contributors_title=escape(_tr("Contributors")),
            empty_text=escape(_tr("No public contributor metadata yet.")),
        )

    rows = []
    for contributor in metadata.contributors:
        name = escape(contributor.name)
        role = escape(localized_contributor_role(contributor.role or "author"))
        lines = [
            f'<div class="contributor-name">{name}</div>',
            f'<span class="role-chip">{role}</span>',
        ]
        if contributor.affiliation:
            lines.append(f'<div class="muted">{escape(contributor.affiliation)}</div>')

        links = []
        if contributor.email:
            email = escape(contributor.email)
            links.append(f'<a class="link-chip" href="mailto:{email}">&#9993; {email}</a>')
        if contributor.url:
            url = escape(contributor.url)
            links.append(f'<a class="link-chip" href="{url}">&#128279; {url}</a>')
        if links:
            lines.append('<div class="links">' + " ".join(links) + "</div>")

        rows.append('<div class="contributor">' + "\n".join(lines) + "</div>")

    return """
    <div class="section">
      <h3>{contributors_title}</h3>
      {rows}
    </div>
    """.format(
        contributors_title=escape(_tr("Contributors")),
        rows="\n".join(rows),
    )


def metadata_html(metadata: CardMetadata) -> str:
    """Render card metadata as compact HTML."""
    card_name = localized_card_name(metadata)
    description_text = localized_card_description(metadata)
    group_text = localized_card_group(metadata)
    fields = [
        (_tr("Class"), metadata.class_name),
        (_tr("Group"), group_text),
        (_tr("Version"), metadata.version),
        (_tr("Maintainer"), metadata.maintainer),
        (_tr("License"), metadata.license),
        (_tr("Source"), _source_label(metadata)),
        (_tr("Source path"), metadata.source_path),
    ]

    chips = [f'<span class="chip chip-source">{escape(_source_label(metadata))}</span>']
    if group_text:
        chips.append(f'<span class="chip chip-group">{escape(group_text)}</span>')
    if metadata.version:
        chips.append(f'<span class="chip chip-version">v{escape(metadata.version)}</span>')

    description = (
        f'<p class="description">{escape(description_text)}</p>'
        if description_text
        else f'<p class="description muted">{escape(_tr("No description provided."))}</p>'
    )

    html = [
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {
  margin: 0;
  background: #f4f6fb;
  color: #1f2937;
  font-family: "Segoe UI", "Microsoft YaHei UI", Arial, sans-serif;
  font-size: 13px;
}
.page {
  padding: 20px;
}

/* ---------- Hero ---------- */
.hero {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #4f46e5;
  border-radius: 12px;
  padding: 22px 24px;
  margin-bottom: 14px;
}
.eyebrow {
  color: #6366f1;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1.2px;
  text-transform: uppercase;
}
h1 {
  margin: 8px 0 10px 0;
  color: #0f172a;
  font-size: 26px;
  font-weight: 700;
  letter-spacing: 0;
}
.description {
  margin: 0 0 14px 0;
  line-height: 1.65;
  color: #334155;
  font-size: 13.5px;
}

/* ---------- Sections ---------- */
.section {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 14px;
}
h3 {
  margin: 0 0 14px 0;
  color: #0f172a;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  border-bottom: 1px solid #eef2f7;
  padding-bottom: 9px;
}

/* ---------- Chips ---------- */
.chip, .role-chip, .link-chip {
  display: inline-block;
  border-radius: 999px;
  padding: 4px 12px;
  margin: 2px 4px 2px 0;
  text-decoration: none;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid transparent;
}
.chip-source {
  background: #eef2ff;
  color: #4338ca;
  border-color: #e0e7ff;
}
.chip-group {
  background: #fef3c7;
  color: #92400e;
  border-color: #fde68a;
}
.chip-version {
  background: #ecfeff;
  color: #155e75;
  border-color: #cffafe;
  font-family: "Consolas", "Courier New", monospace;
}
.role-chip {
  background: #f1f5f9;
  color: #475569;
  border-color: #e2e8f0;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  padding: 3px 10px;
}
.link-chip {
  background: #ecfdf5;
  color: #047857;
  border-color: #d1fae5;
  font-size: 12px;
}

/* ---------- Contributors ---------- */
.contributor {
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 10px;
  background: #f8fafc;
}
.contributor-name {
  color: #0f172a;
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 6px;
}
.links {
  margin-top: 8px;
}

/* ---------- Metadata table ---------- */
.meta-table {
  width: 100%;
  border-collapse: collapse;
}
.meta-table td {
  border-bottom: 1px solid #f1f5f9;
  padding: 10px 6px;
  vertical-align: top;
}
.meta-table tr:last-child td {
  border-bottom: 0;
}
.key {
  width: 130px;
  color: #94a3b8;
  font-weight: 700;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.value {
  color: #1f2937;
}
.value a {
  color: #4f46e5;
  text-decoration: none;
  font-weight: 500;
}

/* ---------- Misc ---------- */
.muted, .empty {
  color: #94a3b8;
  font-style: italic;
}
.citation {
  line-height: 1.7;
  white-space: pre-wrap;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3px solid #4f46e5;
  border-radius: 8px;
  padding: 14px 16px;
  font-family: "Consolas", "Courier New", monospace;
  font-size: 12px;
  color: #334155;
}
</style>
</head>
<body>
<div class="page">
""",
        """
<div class="hero">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{name}</h1>
  {description}
  <div>{chips}</div>
</div>
""".format(
            name=escape(card_name),
            description=description,
            chips=" ".join(chips),
            eyebrow=escape(_tr("Make Dataset Card")),
        ),
    ]
    html.append(_contributors_html(metadata))
    rows = []
    for key, value in fields:
        if value:
            rows.append(
                "<tr>"
                f'<td class="key">{escape(key)}</td>'
                f'<td class="value">{escape(str(value))}</td>'
                "</tr>"
            )
    if metadata.docs_url:
        url = escape(metadata.docs_url)
        rows.append(
            "<tr>"
            f'<td class="key">{escape(_tr("Docs"))}</td>'
            f'<td class="value"><a href="{url}">{url}</a></td>'
            "</tr>"
        )
    if rows:
        html.append(
            """
<div class="section">
  <h3>{metadata_title}</h3>
  <table class="meta-table">
    {rows}
  </table>
</div>
""".format(
                metadata_title=escape(_tr("Metadata")),
                rows="\n".join(rows),
            )
        )
    if metadata.citation:
        html.append(
            """
<div class="section">
  <h3>{citation_title}</h3>
  <div class="citation">{citation}</div>
</div>
""".format(
                citation_title=escape(_tr("Citation")),
                citation=escape(metadata.citation),
            )
        )
    html.append("</div></body></html>")
    return "\n".join(html)


class CardMetadataDialog(QDialog):
    """Dialog showing metadata for one card."""

    def __init__(self, metadata: CardMetadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Card info - {card_name}").format(card_name=metadata.card_name))
        self.resize(580, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(8)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            "QTextBrowser { border: 0; background: #f4f6fb; }"
        )
        browser.setHtml(metadata_html(metadata))
        layout.addWidget(browser)

        close_button = QPushButton(self.tr("Close"), self)
        close_button.setMinimumWidth(96)
        close_button.clicked.connect(self.accept)
        layout.addWidget(
            close_button, alignment=Qt.AlignmentFlag.AlignRight
        )


class CardLibraryDialog(QDialog):
    """Dialog listing all registered cards and their public metadata."""

    cardRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Card library"))
        self.resize(880, 560)
        self._metadata_by_class = dict(CardManager.card_metadata_dict)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel(self.tr("Make Dataset cards"), self)
        title.setStyleSheet(
            "QLabel { color: #0f172a; font-size: 16px; font-weight: 700; "
            "padding: 4px 2px; }"
        )
        root.addWidget(title)

        self.search_edit = QLineEdit(self)
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setPlaceholderText(
            self.tr("Search by card name, group, or description")
        )
        self.search_edit.textChanged.connect(self._filter_cards)
        root.addWidget(self.search_edit)

        body = QHBoxLayout()
        body.setSpacing(10)

        self.card_list = QListWidget(self)
        self.card_list.setMinimumWidth(280)
        self.card_list.setStyleSheet(
            "QListWidget {"
            "  background: #ffffff;"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 8px;"
            "  padding: 4px;"
            "  outline: 0;"
            "}"
            "QListWidget::item {"
            "  padding: 8px 10px;"
            "  border-radius: 6px;"
            "  color: #1f2937;"
            "}"
            "QListWidget::item:selected {"
            "  background: #eef2ff;"
            "  color: #4338ca;"
            "}"
            "QListWidget::item:hover {"
            "  background: #f1f5f9;"
            "}"
        )

        self.detail = QTextBrowser(self)
        self.detail.setOpenExternalLinks(True)
        self.detail.setStyleSheet(
            "QTextBrowser {"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 8px;"
            "  background: #f4f6fb;"
            "}"
        )
        body.addWidget(self.card_list, 1)
        body.addWidget(self.detail, 2)
        root.addLayout(body)

        footer = QHBoxLayout()
        self.result_count_label = QLabel(self)
        footer.addWidget(self.result_count_label)
        footer.addStretch(1)
        self.add_button = QPushButton(self.tr("Add selected card"), self)
        self.add_button.setMinimumWidth(128)
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self._add_current_card)
        footer.addWidget(self.add_button)
        close_button = QPushButton(self.tr("Close"), self)
        close_button.setMinimumWidth(96)
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        root.addLayout(footer)

        for class_name, metadata in sorted(
            self._metadata_by_class.items(),
            key=lambda item: ((item[1].group or ""), item[1].card_name),
        ):
            group = localized_card_group(metadata)
            prefix = f"[{group}]  " if group else ""
            item = QListWidgetItem(f"{prefix}{localized_card_name(metadata)}")
            item.setData(Qt.ItemDataRole.UserRole, class_name)
            item.setToolTip(card_tooltip(metadata))
            self.card_list.addItem(item)

        self.card_list.currentItemChanged.connect(self._show_item)
        self.card_list.itemDoubleClicked.connect(
            lambda _item: self._add_current_card()
        )
        if self.card_list.count():
            self.card_list.setCurrentRow(0)
        self._update_result_count()

    def _show_item(self, item, _previous=None):
        if item is None:
            self.detail.clear()
            self.add_button.setEnabled(False)
            return
        class_name = item.data(Qt.ItemDataRole.UserRole)
        metadata = self._metadata_by_class.get(class_name)
        if metadata is None:
            self.detail.clear()
            self.add_button.setEnabled(False)
            return
        self.detail.setHtml(metadata_html(metadata))
        self.add_button.setEnabled(True)

    def _filter_cards(self, text: str) -> None:
        """Filter the library across user-facing card metadata."""
        query = text.strip().casefold()
        first_visible = None
        for row in range(self.card_list.count()):
            item = self.card_list.item(row)
            class_name = item.data(Qt.ItemDataRole.UserRole)
            metadata = self._metadata_by_class.get(class_name)
            searchable = " ".join(
                (
                    class_name or "",
                    getattr(metadata, "card_name", ""),
                    getattr(metadata, "group", "") or "",
                    getattr(metadata, "description", "") or "",
                    localized_card_name(metadata),
                    localized_card_group(metadata),
                    localized_card_description(metadata),
                )
            ).casefold()
            item.setHidden(bool(query and query not in searchable))
            if not item.isHidden() and first_visible is None:
                first_visible = item
        self.card_list.setCurrentItem(first_visible)
        self._update_result_count()

    def _update_result_count(self) -> None:
        visible_count = sum(
            not self.card_list.item(row).isHidden()
            for row in range(self.card_list.count())
        )
        self.result_count_label.setText(
            self.tr("{count} cards").format(count=visible_count)
        )

    def _add_current_card(self) -> None:
        item = self.card_list.currentItem()
        if item is None or item.isHidden():
            return
        class_name = item.data(Qt.ItemDataRole.UserRole)
        if class_name:
            self.cardRequested.emit(class_name)
