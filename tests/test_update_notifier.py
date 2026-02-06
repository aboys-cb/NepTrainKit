from __future__ import annotations

from typing import Any

import pytest

from NepTrainKit.ui.update import (
    build_compact_summary,
    build_cumulative_release_notes,
    collect_newer_releases,
    fetch_releases,
    is_newer_version,
    normalize_tag_version,
    should_notify_version,
    should_run_auto_check,
)


def test_normalize_tag_version_removes_v_prefix() -> None:
    assert normalize_tag_version("v2.6.3") == "2.6.3"
    assert normalize_tag_version("V2.6.3b1") == "2.6.3b1"


def test_is_newer_version_handles_beta_and_stable() -> None:
    assert is_newer_version("2.6.3", "2.6.1") is True
    assert is_newer_version("2.6.3", "2.6.3b1") is True
    assert is_newer_version("2.6.3b1", "2.6.3") is False


def test_collect_newer_releases_filters_and_sorts() -> None:
    releases = [
        {
            "tag_name": "v2.6.3",
            "name": "v2.6.3",
            "body": "new",
            "published_at": "2025-09-14T00:00:00Z",
            "draft": False,
            "prerelease": False,
        },
        {
            "tag_name": "v2.6.1",
            "name": "v2.6.1",
            "body": "mid",
            "published_at": "2025-09-12T00:00:00Z",
            "draft": False,
            "prerelease": False,
        },
        {
            "tag_name": "v2.6.4b1",
            "name": "v2.6.4b1",
            "body": "beta",
            "published_at": "2025-09-15T00:00:00Z",
            "draft": False,
            "prerelease": True,
        },
        {
            "tag_name": "v2.6.0",
            "name": "v2.6.0",
            "body": "old",
            "published_at": "2025-09-10T00:00:00Z",
            "draft": False,
            "prerelease": False,
        },
    ]

    newer = collect_newer_releases("2.6.0", releases)
    assert [item["tag_name"] for item in newer] == ["v2.6.1", "v2.6.3"]


def test_build_cumulative_release_notes_contains_all_releases() -> None:
    releases = [
        {
            "tag_name": "v2.6.1",
            "name": "v2.6.1",
            "body": "First change",
            "published_at": "2025-09-12T00:00:00Z",
        },
        {
            "tag_name": "v2.6.3",
            "name": "v2.6.3",
            "body": "Second change",
            "published_at": "2025-09-14T00:00:00Z",
        },
    ]

    notes = build_cumulative_release_notes(releases)
    assert "v2.6.1" in notes
    assert "v2.6.3" in notes
    assert "First change" in notes
    assert "Second change" in notes


def test_build_compact_summary_truncates_long_text() -> None:
    notes = "A" * 400
    summary = build_compact_summary(notes, max_length=100)
    assert len(summary) <= 100
    assert summary.endswith("...")


def test_should_run_auto_check_uses_interval() -> None:
    now = 1_700_000_000
    assert should_run_auto_check(0, now, interval_seconds=86400) is True
    assert should_run_auto_check(now - 100, now, interval_seconds=86400) is False
    assert should_run_auto_check(now - 86400, now, interval_seconds=86400) is True


def test_should_notify_version_avoids_duplicate_notify() -> None:
    assert should_notify_version("", "2.6.3") is True
    assert should_notify_version("2.6.3", "2.6.3") is False
    assert should_notify_version("2.6.2", "2.6.3") is True
    assert should_notify_version("2.6.4", "2.6.3") is False


class _MockResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def test_fetch_releases_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    def _mock_get(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(
            200,
            [
                {
                    "tag_name": "v2.6.3",
                    "name": "v2.6.3",
                    "body": "ok",
                }
            ],
        )

    monkeypatch.setattr(requests, "get", _mock_get)
    data = fetch_releases()
    assert isinstance(data, list)
    assert data[0]["tag_name"] == "v2.6.3"


def test_fetch_releases_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    def _mock_get(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(403, {"message": "rate limit exceeded"})

    monkeypatch.setattr(requests, "get", _mock_get)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        fetch_releases()
