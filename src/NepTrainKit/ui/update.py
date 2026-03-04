#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qt helpers for checking NepTrainKit updates and downloading assets."""

from __future__ import annotations

import json
import re
import time
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from loguru import logger
from qfluentwidgets import CaptionLabel, MessageBox, MessageBoxBase, TextEdit

from NepTrainKit import is_nuitka_compiled, module_path
from NepTrainKit.config import Config
from NepTrainKit.core import MessageManager
from NepTrainKit.version import RELEASES_LIST_API_URL, RELEASES_URL, __version__
from NepTrainKit.ui.threads import LoadingThread

AUTO_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
RELEASES_PER_PAGE = 30
REQUEST_TIMEOUT: tuple[float, float] = (2.0, 5.0)
MAX_CACHED_NOTES_LENGTH = 12_000
UPDATE_SECTION = "update"


def normalize_tag_version(tag: str) -> str:
    """Normalize release tag text to a version string."""
    value = str(tag or "").strip()
    if value.lower().startswith("v"):
        value = value[1:]
    return value.strip()


def _fallback_version_key(version: str) -> tuple[int, int, int, int, int]:
    """Build a comparable key for versions when ``packaging`` is unavailable."""
    normalized = normalize_tag_version(version).lower()
    pre_match = re.search(r"(?:b|beta)(\d+)$", normalized)
    main = normalized[: pre_match.start()] if pre_match else normalized
    main = main.rstrip(".-_")
    digits = [int(part) for part in re.findall(r"\d+", main)]
    while len(digits) < 3:
        digits.append(0)
    digits = digits[:3]
    stable_flag = 0 if pre_match else 1
    pre_num = int(pre_match.group(1)) if pre_match else 0
    return digits[0], digits[1], digits[2], stable_flag, pre_num


def _packaging_is_newer(remote: str, local: str) -> bool | None:
    """Try comparing versions via ``packaging.version`` and return ``None`` on fallback."""
    try:
        from packaging.version import InvalidVersion, Version
    except Exception:
        return None

    try:
        return Version(remote) > Version(local)
    except InvalidVersion:
        return None


def is_newer_version(remote: str, local: str) -> bool:
    """Return ``True`` when ``remote`` is newer than ``local``."""
    remote_norm = normalize_tag_version(remote)
    local_norm = normalize_tag_version(local)
    if not remote_norm:
        return False
    if not local_norm:
        return True

    packaging_result = _packaging_is_newer(remote_norm, local_norm)
    if packaging_result is not None:
        return packaging_result

    return _fallback_version_key(remote_norm) > _fallback_version_key(local_norm)


def fetch_releases(timeout: tuple[float, float] = REQUEST_TIMEOUT, per_page: int = RELEASES_PER_PAGE) -> list[dict[str, Any]]:
    """Fetch GitHub releases for NepTrainKit."""
    import requests

    headers = {"User-Agent": "NepTrainKit-UpdateChecker"}
    count = max(1, int(per_page))
    url = f"{RELEASES_LIST_API_URL}?per_page={count}"
    response = requests.get(url, headers=headers, timeout=timeout)
    if response.status_code != 200:
        message = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = str(payload.get("message") or "").strip()
        except Exception:
            message = ""
        detail = f": {message}" if message else ""
        raise RuntimeError(f"Update check failed with HTTP {response.status_code}{detail}")

    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected releases payload from GitHub API.")

    return [item for item in payload if isinstance(item, dict)]


def collect_newer_releases(current_version: str, releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return releases newer than ``current_version`` sorted from old to new."""
    newer: list[dict[str, Any]] = []
    for release in releases:
        if bool(release.get("draft")) or bool(release.get("prerelease")):
            continue
        tag = str(release.get("tag_name") or "").strip()
        if not tag:
            continue
        if is_newer_version(tag, current_version):
            newer.append(release)

    newer.sort(key=lambda item: str(item.get("published_at") or ""))
    return newer


def build_cumulative_release_notes(releases: list[dict[str, Any]]) -> str:
    """Build release notes text covering all provided releases."""
    if not releases:
        return ""

    chunks: list[str] = []
    for release in releases:
        tag = str(release.get("tag_name") or "").strip()
        name = str(release.get("name") or tag or "Release").strip()
        published = str(release.get("published_at") or "").strip()
        date_text = published[:10] if len(published) >= 10 else ""
        header = f"### {name} ({tag})" if tag else f"### {name}"
        if date_text:
            header = f"{header} - {date_text}"
        body = str(release.get("body") or "").strip() or "(No release notes provided.)"
        chunks.append(f"{header}\n{body}")

    return "\n\n".join(chunks).strip()


def build_compact_summary(notes: str, max_length: int = 180) -> str:
    """Build a single-line summary string for notifications."""
    compact = re.sub(r"\s+", " ", str(notes or "")).strip()
    if not compact:
        return "See release notes for details."
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def should_run_auto_check(
    last_check_ts: int | str | None,
    now_ts: int | None = None,
    interval_seconds: int = AUTO_CHECK_INTERVAL_SECONDS,
) -> bool:
    """Return whether the auto-update check should run now."""
    now_value = int(now_ts if now_ts is not None else time.time())
    try:
        last_value = int(last_check_ts or 0)
    except Exception:
        last_value = 0
    interval = max(1, int(interval_seconds))
    if last_value <= 0:
        return True
    return (now_value - last_value) >= interval


def should_notify_version(last_notified_version: str, latest_version: str) -> bool:
    """Return whether ``latest_version`` should trigger a new notification."""
    latest = normalize_tag_version(latest_version)
    last = normalize_tag_version(last_notified_version)
    if not latest:
        return False
    if not last:
        return True
    if latest == last:
        return False
    return is_newer_version(latest, last)


def _trim_cached_notes(notes: str, max_len: int = MAX_CACHED_NOTES_LENGTH) -> str:
    """Trim cached notes to a bounded size for local config storage."""
    text = str(notes or "")
    if len(text) <= max_len:
        return text
    suffix = "\n\n[... truncated in local cache ...]"
    allowed = max(0, max_len - len(suffix))
    return text[:allowed].rstrip() + suffix


def _set_pending_update_state(version: str, notes: str, release_url: str) -> None:
    """Store pending update details in config."""
    Config.set(UPDATE_SECTION, "pending_version", normalize_tag_version(version))
    Config.set(UPDATE_SECTION, "pending_notes", _trim_cached_notes(notes))
    Config.set(UPDATE_SECTION, "pending_release_url", str(release_url or RELEASES_URL))


def clear_pending_update_state() -> None:
    """Clear pending update details from config."""
    Config.set(UPDATE_SECTION, "pending_version", "")
    Config.set(UPDATE_SECTION, "pending_notes", "")
    Config.set(UPDATE_SECTION, "pending_release_url", "")


def get_pending_update_version() -> str:
    """Get cached pending version when it is still newer than local version."""
    pending = normalize_tag_version(str(Config.get(UPDATE_SECTION, "pending_version", "") or "").strip())
    if not pending:
        return ""
    if is_newer_version(pending, __version__):
        return pending
    clear_pending_update_state()
    return ""


class ReleaseNotesMessageBox(MessageBoxBase):
    """Dialog that displays full cumulative release notes."""

    def __init__(self, title: str, notes: str, parent=None):
        super().__init__(parent)
        self.headerLabel = CaptionLabel(title, self)
        self.headerLabel.setWordWrap(True)
        self.notesEdit = TextEdit(self)
        self.notesEdit.setReadOnly(True)
        self.notesEdit.setPlainText(notes)
        self.notesEdit.setMinimumHeight(360)
        self.viewLayout.addWidget(self.headerLabel)
        self.viewLayout.addWidget(self.notesEdit)
        self.widget.setMinimumWidth(760)


class UpdateWoker(QObject):
    """Worker that checks for new NepTrainKit releases."""

    result = Signal(dict)

    def __init__(self, parent):
        self._parent = parent
        super().__init__(parent)
        self._manual = True
        self._on_finished: Callable[[dict[str, Any]], None] | None = None
        self.result.connect(self._check_update_call_back)
        self.update_thread = LoadingThread(self._parent, show_tip=False)

    def _check_update(self) -> None:
        """Query GitHub releases and emit a normalized result payload."""
        try:
            releases = fetch_releases(timeout=REQUEST_TIMEOUT, per_page=RELEASES_PER_PAGE)
            newer_releases = collect_newer_releases(__version__, releases)
            if not newer_releases:
                self.result.emit({"ok": True, "has_update": False})
                return

            latest_release = newer_releases[-1]
            latest_version = normalize_tag_version(str(latest_release.get("tag_name") or ""))
            notes = build_cumulative_release_notes(newer_releases)
            release_url = str(latest_release.get("html_url") or RELEASES_URL)
            self.result.emit(
                {
                    "ok": True,
                    "has_update": True,
                    "latest_version": latest_version,
                    "notes": notes,
                    "summary": build_compact_summary(notes),
                    "release_url": release_url,
                }
            )
        except Exception as exc:
            logger.error(traceback.format_exc())
            self.result.emit({"ok": False, "has_update": False, "error": str(exc)})

    def _check_update_call_back(self, result: dict[str, Any]) -> None:
        """Handle check results in UI thread for manual requests."""
        def _finish() -> None:
            if self._on_finished is not None:
                try:
                    self._on_finished(result)
                except Exception:
                    logger.error(traceback.format_exc())

        if not self._manual:
            _finish()
            return

        if not result.get("ok"):
            error = str(result.get("error") or "Network error!")
            MessageManager.send_error_message(error, title="Update Check Failed")
            _finish()
            return

        if not result.get("has_update"):
            clear_pending_update_state()
            MessageManager.send_success_message("You are already using the latest version!")
            _finish()
            return

        latest_version = str(result.get("latest_version") or "").strip()
        notes = str(result.get("notes") or "").strip()
        release_url = str(result.get("release_url") or RELEASES_URL).strip() or RELEASES_URL
        _set_pending_update_state(latest_version, notes, release_url)

        title = f"New version available: v{latest_version}" if latest_version else "New version available"
        box = ReleaseNotesMessageBox(title, notes, self._parent)
        box.yesButton.setText("Open Releases")
        box.cancelButton.setText("Close")
        box.exec_()
        if box.result() != 0:
            QDesktopServices.openUrl(QUrl(release_url))

        if not is_nuitka_compiled:
            MessageManager.send_info_message(
                "Upgrade command: python -m pip install -U --pre NepTrainKit",
                title="Pip Upgrade",
            )
        _finish()

    def check_update(
        self,
        manual: bool = True,
        on_finished: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Start asynchronous release check.

        Parameters
        ----------
        manual : bool, default=True
            ``True`` to show user-facing dialogs/messages.
        on_finished : callable | None
            Optional callback receiving result payload.
        """
        self._manual = manual
        self._on_finished = on_finished
        if self.update_thread.isRunning():
            return
        if manual:
            MessageManager.send_info_message("Checking for updates, please wait...")
        self.update_thread.start_work(self._check_update)


class AutoUpdateNotifier(QObject):
    """Coordinator for startup auto-checks and non-blocking notifications."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.update_worker = UpdateWoker(parent)
        self._startup_notice_sent = False
        self._startup_notice_version = ""

    def start_if_due(self) -> None:
        """Run the auto-check only when daily interval has elapsed."""
        self._show_startup_pending_notice()
        now = int(time.time())
        last_check_ts = Config.getint(UPDATE_SECTION, "last_auto_check_ts", 0) or 0
        if not should_run_auto_check(last_check_ts, now, AUTO_CHECK_INTERVAL_SECONDS):
            return
        self.update_worker.check_update(manual=False, on_finished=self._handle_auto_result)

    def _show_startup_pending_notice(self) -> None:
        """Show one non-blocking reminder per startup when a pending update exists."""
        if self._startup_notice_sent:
            return
        pending_version = get_pending_update_version()
        if not pending_version:
            return
        notice = f"New version v{pending_version} is available. Open Settings > About > Check for Updates for details."
        MessageManager.send_info_message(notice, title="Update available")
        self._startup_notice_sent = True
        self._startup_notice_version = pending_version

    def _handle_auto_result(self, result: dict[str, Any]) -> None:
        """Process auto-check result without blocking UI."""
        def _refresh_parent_indicators() -> None:
            if hasattr(self._parent, "refresh_update_indicators"):
                try:
                    self._parent.refresh_update_indicators()
                except Exception:
                    logger.error(traceback.format_exc())

        now = int(time.time())
        Config.set(UPDATE_SECTION, "last_auto_check_ts", now)

        if not result.get("ok"):
            logger.warning(f"Automatic update check failed: {result.get('error', 'unknown error')}")
            _refresh_parent_indicators()
            return

        if not result.get("has_update"):
            clear_pending_update_state()
            _refresh_parent_indicators()
            return

        latest_version = str(result.get("latest_version") or "").strip()
        notes = str(result.get("notes") or "")
        release_url = str(result.get("release_url") or RELEASES_URL).strip() or RELEASES_URL
        _set_pending_update_state(latest_version, notes, release_url)
        _refresh_parent_indicators()

        last_notified = str(Config.get(UPDATE_SECTION, "last_notified_version", "") or "").strip()
        if not should_notify_version(last_notified, latest_version):
            return

        summary = str(result.get("summary") or "")
        if self._startup_notice_sent and normalize_tag_version(latest_version) == normalize_tag_version(self._startup_notice_version):
            Config.set(UPDATE_SECTION, "last_notified_version", latest_version)
            return

        notice = f"New version v{latest_version} is available. {summary}".strip()
        notice = build_compact_summary(notice, max_length=220)
        notice += " Open Settings > About > Check for Updates for details."
        MessageManager.send_info_message(notice, title="Update available")
        Config.set(UPDATE_SECTION, "last_notified_version", latest_version)


class UpdateNEP89Woker(QObject):
    """Worker that keeps the bundled ``nep89`` potential file up to date."""

    version = Signal(int)
    download_success = Signal()

    def __init__(self, parent):
        """Initialise the worker with the owning ``parent`` widget."""
        self._parent = parent
        super().__init__(parent)
        self.func = self._check_update
        self.version.connect(self._check_update_call_back)
        self.update_thread = LoadingThread(self._parent, show_tip=False)
        self.down_thread = LoadingThread(self._parent, show_tip=True, title="Downloading")

    def download(self, latest_date: int) -> None:
        """Download the latest ``nep89`` model and refresh metadata."""
        import requests

        raw_url = (
            "https://raw.githubusercontent.com/brucefan1983/GPUMD/master/"
            f"potentials/nep/nep89_{latest_date}/nep89_{latest_date}.txt"
        )
        response = requests.get(raw_url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        nep89_path = module_path / "Config/nep89.txt"
        with nep89_path.open("wb") as target:
            for chunk in response.iter_content(1024):
                if chunk:
                    target.write(chunk)

        MessageManager.send_success_message("Update large model completed!")
        nep_json_path = module_path / "Config/nep.json"
        with nep_json_path.open("r", encoding="utf-8") as config_file:
            local_nep_info = json.load(config_file)
        local_nep_info["date"] = latest_date
        with nep_json_path.open("w", encoding="utf-8") as config_file:
            json.dump(local_nep_info, config_file)

    def _check_update(self) -> None:
        """Check the remote repository for a newer ``nep89`` dataset."""
        import requests

        MessageManager.send_info_message("Checking for updates, please wait...")
        api_url = "https://api.github.com/repos/brucefan1983/GPUMD/contents/potentials/nep"
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            MessageManager.send_warning_message(
                f"Unable to access the warehouse directory, status code: {response.status_code}"
            )
            return
        directories = [
            item["name"]
            for item in response.json()
            if item["type"] == "dir" and item["name"].startswith("nep89_")
        ]

        date_pattern = re.compile(r"nep89_(\d{8})")
        latest_date: int | None = None
        for dir_name in directories:
            match = date_pattern.match(dir_name)
            if match:
                current_date = int(match.group(1))
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date

        if latest_date is None:
            MessageManager.send_warning_message("No NEP89 release directory found in upstream repository.")
            return

        self.version.emit(latest_date)

    def _check_update_call_back(self, latest_date: int) -> None:
        """Prompt the user to download the updated ``nep89`` archive."""
        nep_json_path = module_path / "Config/nep.json"
        with nep_json_path.open("r", encoding="utf-8") as config_file:
            local_nep_info = json.load(config_file)
        if local_nep_info["date"] >= latest_date:
            MessageManager.send_success_message("You are already using the latest version!")
            return
        box = MessageBox(
            "New version",
            f"A new version of the large model has been detected:{latest_date}",
            self._parent,
        )
        box.yesButton.setText("Update")
        box.cancelButton.setText("Cancel")
        box.exec_()
        if box.result() == 0:
            return
        self.down_thread.start_work(self.download, latest_date)

    def check_update(self) -> None:
        """Start checking for ``nep89`` updates in the background."""
        self.update_thread.start_work(self._check_update)


__all__ = [
    "AUTO_CHECK_INTERVAL_SECONDS",
    "AutoUpdateNotifier",
    "UpdateWoker",
    "UpdateNEP89Woker",
    "build_compact_summary",
    "build_cumulative_release_notes",
    "clear_pending_update_state",
    "collect_newer_releases",
    "fetch_releases",
    "get_pending_update_version",
    "is_newer_version",
    "normalize_tag_version",
    "should_notify_version",
    "should_run_auto_check",
]
