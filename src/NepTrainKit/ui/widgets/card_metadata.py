"""Dialogs and formatting helpers for Make Dataset card metadata."""

from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from NepTrainKit.core import CardManager, CardMetadata


def contributor_label(contributor) -> str:
    """Return a compact public label for one contributor."""
    role = f" ({contributor.role})" if contributor.role else ""
    return f"{contributor.name}{role}"


def contributors_text(metadata: CardMetadata) -> str:
    """Return contributor names for plain-text UI surfaces."""
    if not metadata.contributors:
        return "Not specified"
    return ", ".join(contributor_label(item) for item in metadata.contributors)


def card_tooltip(metadata: CardMetadata) -> str:
    """Build a short tooltip for an Add Card action."""
    lines = [metadata.card_name]
    if metadata.description:
        lines.append(metadata.description)
    lines.append(f"Contributors: {contributors_text(metadata)}")
    if metadata.version:
        lines.append(f"Version: {metadata.version}")
    return "\n".join(lines)


def _source_label(metadata: CardMetadata) -> str:
    path = Path(metadata.source_path) if metadata.source_path else None
    if path and path.parent.name == "_card":
        return "Built-in"
    if path:
        return "Custom"
    return "Unknown"


def _contributors_html(metadata: CardMetadata) -> str:
    if not metadata.contributors:
        return """
        <div class="section">
          <h3>Contributors</h3>
          <p class="empty">No public contributor metadata yet.</p>
        </div>
        """

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
      <h3>Contributors</h3>
      {rows}
    </div>
    """.format(rows="\n".join(rows))


def metadata_html(metadata: CardMetadata) -> str:
    """Render card metadata as compact HTML."""
    fields = [
        ("Class", metadata.class_name),
        ("Group", metadata.group or ""),
        ("Version", metadata.version),
        ("Maintainer", metadata.maintainer),
        ("License", metadata.license),
        ("Source", _source_label(metadata)),
        ("Source path", metadata.source_path),
    ]

    chips = [f'<span class="chip chip-source">{escape(_source_label(metadata))}</span>']
    if metadata.group:
        chips.append(f'<span class="chip chip-group">{escape(metadata.group)}</span>')
    if metadata.version:
        chips.append(f'<span class="chip chip-version">v{escape(metadata.version)}</span>')

    description = (
        f'<p class="description">{escape(metadata.description)}</p>'
        if metadata.description
        else '<p class="description muted">No description provided.</p>'
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
  <div class="eyebrow">Make Dataset Card</div>
  <h1>{name}</h1>
  {description}
  <div>{chips}</div>
</div>
""".format(
            name=escape(metadata.card_name),
            description=description,
            chips=" ".join(chips),
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
            '<td class="key">Docs</td>'
            f'<td class="value"><a href="{url}">{url}</a></td>'
            "</tr>"
        )
    if rows:
        html.append(
            """
<div class="section">
  <h3>Metadata</h3>
  <table class="meta-table">
    {rows}
  </table>
</div>
""".format(rows="\n".join(rows))
        )
    if metadata.citation:
        html.append(
            """
<div class="section">
  <h3>Citation</h3>
  <div class="citation">{citation}</div>
</div>
""".format(citation=escape(metadata.citation))
        )
    html.append("</div></body></html>")
    return "\n".join(html)


class CardMetadataDialog(QDialog):
    """Dialog showing metadata for one card."""

    def __init__(self, metadata: CardMetadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Card Info - {metadata.card_name}")
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

        close_button = QPushButton("Close", self)
        close_button.setMinimumWidth(96)
        close_button.clicked.connect(self.accept)
        layout.addWidget(
            close_button, alignment=Qt.AlignmentFlag.AlignRight
        )


class CardLibraryDialog(QDialog):
    """Dialog listing all registered cards and their public metadata."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Card Library")
        self.resize(880, 560)
        self._metadata_by_class = dict(CardManager.card_metadata_dict)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Make Dataset Cards", self)
        title.setStyleSheet(
            "QLabel { color: #0f172a; font-size: 16px; font-weight: 700; "
            "padding: 4px 2px; }"
        )
        root.addWidget(title)

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

        close_button = QPushButton("Close", self)
        close_button.setMinimumWidth(96)
        close_button.clicked.connect(self.accept)
        root.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

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
        if self.card_list.count():
            self.card_list.setCurrentRow(0)

    def _show_item(self, item, _previous=None):
        if item is None:
            self.detail.clear()
            return
        class_name = item.data(Qt.ItemDataRole.UserRole)
        metadata = self._metadata_by_class.get(class_name)
        if metadata is None:
            self.detail.clear()
            return
        self.detail.setHtml(metadata_html(metadata))
