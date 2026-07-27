"""Dialogs and formatting helpers for Make Dataset card metadata."""

from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
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
            QCoreApplication.translate("CardCatalog", "Branch Merge Group"),
            QCoreApplication.translate(
                "CardCatalog",
                "Run several independent child branches one at a time from the same input, merge their outputs, and optionally apply one post-filter; child cards do not feed one another.",
            ),
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
            QCoreApplication.translate(
                "CardCatalog",
                "Build a one-dimensional composition transition along lattice a, b, or c without moving atoms.",
            ),
        ),
        "CompositionSweepCard": (
            QCoreApplication.translate("CardCatalog", "Composition Space Sampling"),
            QCoreApplication.translate(
                "CardCatalog",
                "Sample target ratios in binary-to-quinary composition spaces. This card only writes Comp(...) tags; add Random Occupancy to change atoms.",
            ),
        ),
        "ConditionalReplaceCard": (
            QCoreApplication.translate("CardCatalog", "Conditional Replace"),
            QCoreApplication.translate(
                "CardCatalog",
                "Select a target element by Cartesian coordinates, then replace every matching site using the specified replacement mixture.",
            ),
        ),
        "CorrelatedRandomSpinCard": (
            QCoreApplication.translate("CardCatalog", "Correlated Random Spin"),
            QCoreApplication.translate("CardCatalog", "Generate non-collinear random spins with an explicit spatial correlation length."),
        ),
        "CrystalPrototypeBuilderCard": (
            QCoreApplication.translate("CardCatalog", "Crystal Prototype Builder"),
            QCoreApplication.translate("CardCatalog", "Generate simple bulk crystal prototypes without requiring input structures."),
        ),
        "FiniteCellAlloyOccupancyCard": (
            QCoreApplication.translate("CardCatalog", "Finite-Cell Alloy Occupancy"),
            QCoreApplication.translate(
                "CardCatalog",
                "Generate real alloy occupancies from integer compositions achievable in the finite cell, with optional independent sublattice constraints.",
            ),
        ),
        "FPSFilterDataCard": (
            QCoreApplication.translate("CardCatalog", "Representative Sampling (FPS)"),
            QCoreApplication.translate(
                "CardCatalog",
                "Keep a descriptor-space representative subset using a chosen NEP model, with either one global budget or guaranteed quotas for every element set; an existing training set can seed coverage.",
            ),
        ),
        "FoldedHelixCard": (
            QCoreApplication.translate("CardCatalog", "Folded Helix"),
            QCoreApplication.translate("CardCatalog", "Assign symmetric clockwise-then-counterclockwise layered helix moments."),
        ),
        "GeometryFilterCard": (
            QCoreApplication.translate("CardCatalog", "Geometry Sanity Filter"),
            QCoreApplication.translate(
                "CardCatalog",
                "Reject empty or non-finite structures and optionally enforce a chemistry-independent shortest-pair cutoff plus bulk cell, volume-per-atom, and mass-density limits.",
            ),
        ),
        "GroupLabelCard": (
            QCoreApplication.translate("CardCatalog", "Group Label"),
            QCoreApplication.translate(
                "CardCatalog",
                "Divide atoms into two coordinate-based groups for downstream magnetic, doping, or vacancy operations; coordinates and elements are unchanged.",
            ),
        ),
        "InsertDefectCard": (
            QCoreApplication.translate(
                "CardCatalog",
                "Interstitial and Surface Adsorption",
            ),
            QCoreApplication.translate(
                "CardCatalog",
                "Sample random interstitial candidates inside a cell or random adsorbates above a selected upper surface; only a minimum-distance constraint is enforced.",
            ),
        ),
        "LayerCopyCard": (
            QCoreApplication.translate("CardCatalog", "Layer Stack"),
            QCoreApplication.translate(
                "CardCatalog",
                "Build one multilayer structure by translating the complete input slab along Cartesian z; an optional dz=f(x,y,z) expression can warp selected atoms before every full-slab copy.",
            ),
        ),
        "LocalSolvationCard": (
            QCoreApplication.translate("CardCatalog", "Local Solvation"),
            QCoreApplication.translate(
                "CardCatalog",
                "Insert solvent molecules around selected host atoms using a fallback COM shell or ion-specific first-shell distances, with collision checks and optional flexible-solvent sampling.",
            ),
        ),
        "MagneticMomentRotationCard": (
            QCoreApplication.translate("CardCatalog", "Magmom Rotation"),
            QCoreApplication.translate("CardCatalog", "Rotate and optionally rescale atomic magnetic moments for selected species."),
        ),
        "MagneticOrderCard": (
            QCoreApplication.translate("CardCatalog", "Magnetic Order"),
            QCoreApplication.translate(
                "CardCatalog",
                "Generate FM, AFM, and random PM initial spin states from element moments without changing coordinates or elements.",
            ),
        ),
        "OrganicMolConfigPBCCard": (
            QCoreApplication.translate("CardCatalog", "Organic Mol Config"),
            QCoreApplication.translate(
                "CardCatalog",
                "Detect molecular bonds, rotate eligible single-bond subtrees, add optional Gaussian noise, and skip conformers that fail bond-length or clash guards.",
            ),
        ),
        "OrderedAlloyPrototypeCard": (
            QCoreApplication.translate("CardCatalog", "Ordered Alloy Prototype"),
            QCoreApplication.translate(
                "CardCatalog",
                "Generate periodic A1, A2, A3, L12, B2, and L10 prototypes with sublattice labels.",
            ),
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
            QCoreApplication.translate(
                "CardCatalog",
                "Remove randomly selected sites using element, existing group, and count rules; other atomic coordinates stay unchanged.",
            ),
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
            QCoreApplication.translate(
                "CardCatalog",
                "Randomly place solvent molecules throughout an existing periodic cell by a fixed target count or a nominal pure-solvent density estimate, while rejecting short contacts.",
            ),
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
            QCoreApplication.translate("CardCatalog", "Legacy Stacking Fault"),
            QCoreApplication.translate(
                "CardCatalog",
                "Load and reproduce existing workflows that used the old automatic-direction layer shift; this compatibility card is hidden from new-card entry points.",
            ),
        ),
        "StrictGSFEPathCard": (
            QCoreApplication.translate("CardCatalog", "Stacking Fault / GSFE Path"),
            QCoreApplication.translate(
                "CardCatalog",
                "Shift atoms above an interlayer cut along an explicit in-plane direction to generate stacking-fault structures or an unrelaxed GSFE path; the input cell must already be oriented to the fault plane.",
            ),
        ),
        "SuperCellCard": (
            QCoreApplication.translate("CardCatalog", "Super Cell"),
            QCoreApplication.translate("CardCatalog", "Create supercells from fixed scale factors, target lattice lengths, or atom limits."),
        ),
        "VacancyDefectCard": (
            QCoreApplication.translate("CardCatalog", "Global Random Vacancy"),
            QCoreApplication.translate(
                "CardCatalog",
                "Delete sites globally by an overall count or fraction without distinguishing elements; remaining coordinates are unchanged.",
            ),
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
        self.resize(940, 600)
        self.setMinimumSize(720, 500)
        self._metadata_by_class = {
            class_name: metadata
            for class_name, metadata in CardManager.card_metadata_dict.items()
            if getattr(
                CardManager.card_info_dict.get(class_name),
                "discoverable",
                True,
            )
        }

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

        body = QSplitter(Qt.Orientation.Horizontal, self)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(8)

        self.card_list = QListWidget(self)
        self.card_list.setMinimumWidth(250)
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

        self.detail = QFrame(self)
        self.detail.setObjectName("cardDetailPanel")
        self.detail.setMinimumWidth(360)
        self.detail.setStyleSheet(
            "QFrame#cardDetailPanel {"
            "  background: #ffffff;"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 10px;"
            "}"
            "QFrame#cardDetailPanel QLabel { border: 0; background: transparent; }"
            "QLabel#cardGroupChip {"
            "  color: #4338ca;"
            "  background: #eef2ff;"
            "  border-radius: 8px;"
            "  padding: 4px 10px;"
            "  font-size: 12px;"
            "  font-weight: 600;"
            "}"
            "QLabel#cardDetailTitle {"
            "  color: #0f172a;"
            "  font-size: 22px;"
            "  font-weight: 700;"
            "}"
            "QLabel#cardCanonicalName { color: #64748b; font-size: 12px; }"
            "QLabel#cardDescription { color: #334155; font-size: 14px; }"
            "QLabel#cardSectionTitle {"
            "  color: #0f172a;"
            "  font-size: 13px;"
            "  font-weight: 700;"
            "}"
            "QLabel#cardInfoKey { color: #64748b; font-size: 12px; }"
            "QLabel#cardInfoValue { color: #1f2937; font-size: 13px; }"
            "QFrame#cardTechnicalPanel {"
            "  background: #f8fafc;"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 8px;"
            "}"
        )
        detail_layout = QVBoxLayout(self.detail)
        detail_layout.setContentsMargins(22, 20, 22, 18)
        detail_layout.setSpacing(10)

        self.detail_group_label = QLabel(self.detail)
        self.detail_group_label.setObjectName("cardGroupChip")
        self.detail_group_label.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        detail_layout.addWidget(self.detail_group_label)

        self.detail_title_label = QLabel(self.detail)
        self.detail_title_label.setObjectName("cardDetailTitle")
        self.detail_title_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_title_label)

        self.detail_canonical_label = QLabel(self.detail)
        self.detail_canonical_label.setObjectName("cardCanonicalName")
        self.detail_canonical_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_canonical_label)

        self.detail_description_label = QLabel(self.detail)
        self.detail_description_label.setObjectName("cardDescription")
        self.detail_description_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_description_label)

        divider = QFrame(self.detail)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e2e8f0;")
        detail_layout.addWidget(divider)

        about_label = QLabel(self.tr("About"), self.detail)
        about_label.setObjectName("cardSectionTitle")
        detail_layout.addWidget(about_label)

        info_layout = QGridLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setHorizontalSpacing(18)
        info_layout.setVerticalSpacing(9)
        info_layout.setColumnStretch(1, 1)
        self._info_rows = []
        for row, key in enumerate(
            (
                self.tr("Contributors"),
                self.tr("Source"),
                self.tr("Version"),
                self.tr("License"),
                self.tr("Documentation"),
            )
        ):
            key_label = QLabel(key, self.detail)
            key_label.setObjectName("cardInfoKey")
            value_label = QLabel(self.detail)
            value_label.setObjectName("cardInfoValue")
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            info_layout.addWidget(key_label, row, 0, Qt.AlignmentFlag.AlignTop)
            info_layout.addWidget(value_label, row, 1)
            self._info_rows.append((key_label, value_label))
        self.detail_contributors_label = self._info_rows[0][1]
        self.detail_source_label = self._info_rows[1][1]
        self.detail_version_label = self._info_rows[2][1]
        self.detail_license_label = self._info_rows[3][1]
        self.detail_docs_label = self._info_rows[4][1]
        self.detail_docs_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.detail_docs_label.setOpenExternalLinks(True)
        detail_layout.addLayout(info_layout)

        self.detail_technical_button = QPushButton(
            self.tr("Technical details"), self.detail
        )
        self.detail_technical_button.setCheckable(True)
        self.detail_technical_button.setStyleSheet(
            "QPushButton {"
            "  color: #475569;"
            "  background: transparent;"
            "  border: 0;"
            "  padding: 6px 0;"
            "  text-align: left;"
            "}"
            "QPushButton:hover { color: #4338ca; }"
        )
        self.detail_technical_button.toggled.connect(
            self._toggle_technical_details
        )
        detail_layout.addWidget(self.detail_technical_button)

        self.detail_technical_panel = QFrame(self.detail)
        self.detail_technical_panel.setObjectName("cardTechnicalPanel")
        technical_layout = QGridLayout(self.detail_technical_panel)
        technical_layout.setContentsMargins(12, 10, 12, 10)
        technical_layout.setHorizontalSpacing(16)
        technical_layout.setVerticalSpacing(7)
        technical_layout.addWidget(QLabel(self.tr("Class")), 0, 0)
        self.detail_class_value = QLabel(self.detail_technical_panel)
        self.detail_class_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        technical_layout.addWidget(self.detail_class_value, 0, 1)
        technical_layout.addWidget(QLabel(self.tr("Source file")), 1, 0)
        self.detail_path_value = QLabel(self.detail_technical_panel)
        self.detail_path_value.setWordWrap(True)
        self.detail_path_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        technical_layout.addWidget(self.detail_path_value, 1, 1)
        technical_layout.setColumnStretch(1, 1)
        self.detail_technical_panel.hide()
        detail_layout.addWidget(self.detail_technical_panel)
        detail_layout.addStretch(1)

        body.addWidget(self.card_list)
        body.addWidget(self.detail)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([300, 610])
        root.addWidget(body, 1)

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
            self._clear_detail()
            self.add_button.setEnabled(False)
            return
        class_name = item.data(Qt.ItemDataRole.UserRole)
        metadata = self._metadata_by_class.get(class_name)
        if metadata is None:
            self._clear_detail()
            self.add_button.setEnabled(False)
            return
        localized_name = localized_card_name(metadata)
        self.detail_group_label.setText(localized_card_group(metadata))
        self.detail_group_label.setVisible(bool(self.detail_group_label.text()))
        self.detail_title_label.setText(localized_name)
        canonical_name = metadata.card_name if metadata.card_name != localized_name else ""
        self.detail_canonical_label.setText(canonical_name)
        self.detail_canonical_label.setVisible(bool(canonical_name))
        self.detail_description_label.setText(
            localized_card_description(metadata)
            or self.tr("No description provided.")
        )

        values = (
            contributors_text(metadata),
            _source_label(metadata),
            metadata.version,
            metadata.license,
            metadata.docs_url,
        )
        for index, ((key_label, value_label), value) in enumerate(
            zip(self._info_rows, values)
        ):
            if index == 4 and value:
                escaped_url = escape(str(value))
                link_text = escape(self.tr("Open documentation"))
                value_label.setText(
                    f'<a href="{escaped_url}">{link_text}</a>'
                )
            else:
                value_label.setText(str(value or ""))
            key_label.setVisible(bool(value))
            value_label.setVisible(bool(value))

        self.detail_class_value.setText(metadata.class_name)
        source_path = Path(metadata.source_path) if metadata.source_path else None
        self.detail_path_value.setText(
            source_path.name if source_path else self.tr("Not specified")
        )
        full_path = str(source_path) if source_path else ""
        self.detail_path_value.setToolTip(full_path)
        self.detail_technical_button.setChecked(False)
        self.detail_technical_button.setEnabled(True)
        self.add_button.setEnabled(True)

    def _clear_detail(self) -> None:
        for label in (
            self.detail_group_label,
            self.detail_title_label,
            self.detail_canonical_label,
            self.detail_description_label,
            self.detail_class_value,
            self.detail_path_value,
        ):
            label.clear()
        for key_label, value_label in self._info_rows:
            key_label.hide()
            value_label.clear()
            value_label.hide()
        self.detail_technical_button.setChecked(False)
        self.detail_technical_button.setEnabled(False)

    def _toggle_technical_details(self, expanded: bool) -> None:
        self.detail_technical_panel.setVisible(expanded)
        self.detail_technical_button.setText(
            self.tr("Hide technical details")
            if expanded
            else self.tr("Technical details")
        )

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
