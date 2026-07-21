"""Dialogs and formatting helpers for Make Dataset card metadata."""

from __future__ import annotations

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


def contributor_label(contributor) -> str:
    """Return a compact public label for one contributor."""
    role = f" ({contributor.role})" if contributor.role else ""
    return f"{contributor.name}{role}"


def contributors_text(metadata: CardMetadata) -> str:
    """Return contributor names for plain-text UI surfaces."""
    if not metadata.contributors:
        return _tr("Not specified")
    return ", ".join(contributor_label(item) for item in metadata.contributors)


def card_tooltip(metadata: CardMetadata) -> str:
    """Build a short tooltip for an Add Card action."""
    lines = [metadata.card_name]
    if metadata.description:
        lines.append(metadata.description)
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
        role = escape(contributor.role or "author")
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
    fields = [
        (_tr("Class"), metadata.class_name),
        (_tr("Group"), metadata.group or ""),
        (_tr("Version"), metadata.version),
        (_tr("Maintainer"), metadata.maintainer),
        (_tr("License"), metadata.license),
        (_tr("Source"), _source_label(metadata)),
        (_tr("Source path"), metadata.source_path),
    ]

    chips = [f'<span class="chip chip-source">{escape(_source_label(metadata))}</span>']
    if metadata.group:
        chips.append(f'<span class="chip chip-group">{escape(metadata.group)}</span>')
    if metadata.version:
        chips.append(f'<span class="chip chip-version">v{escape(metadata.version)}</span>')

    description = (
        f'<p class="description">{escape(metadata.description)}</p>'
        if metadata.description
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
            name=escape(metadata.card_name),
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
            prefix = f"[{metadata.group}]  " if metadata.group else ""
            item = QListWidgetItem(f"{prefix}{metadata.card_name}")
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
