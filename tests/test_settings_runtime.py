from unittest.mock import patch

from PySide6.QtWidgets import QApplication

import NepTrainKit.ui.pages.settings as settings_module
from NepTrainKit.core.runtime_health import RuntimeCapability, RuntimeHealth
from NepTrainKit.ui.pages.settings import SettingsWidget


def _report(*, native_complete=True, cpu=True, cuda=False) -> RuntimeHealth:
    native = tuple(
        RuntimeCapability(
            name=name,
            available=native_complete or name != "_phase",
            reason="available" if native_complete or name != "_phase" else "ImportError",
        )
        for name in ("_io", "_audit", "_phase", "_magnetism", "_sampling")
    )
    return RuntimeHealth(
        native=native,
        adapters_version="1.2.3",
        cpu=RuntimeCapability("cpu", cpu, "available" if cpu else "module_missing"),
        cuda=RuntimeCapability("cuda", cuda, "available" if cuda else "module_missing"),
    )


def test_settings_runtime_card_summarizes_capabilities():
    QApplication.instance() or QApplication([])
    with patch.object(settings_module, "inspect_runtime_health", return_value=_report()):
        widget = SettingsWidget(None)

    text = widget.runtime_health_card.contentLabel.text()
    assert "5/5" in text
    assert "1.2.3" in text
    assert "CPU: Available" in text
    assert "CUDA: Unavailable" in text
    assert widget.runtime_update_card is not None
    assert "nep-adapters" in widget.runtime_update_card.titleLabel.text()
    assert "nep-adapters" in widget.runtime_update_card.contentLabel.text()
    assert not hasattr(widget, "runtime_group")
    assert widget.runtime_health_card.parent() is widget.nep_group
    assert widget.runtime_update_card.parent() is widget.nep_group


def test_settings_runtime_update_card_reports_restart_requirement():
    QApplication.instance() or QApplication([])
    with patch.object(settings_module, "inspect_runtime_health", return_value=_report()):
        widget = SettingsWidget(None)

    widget._on_runtime_update_finished(
        {"ok": True, "updated": True, "version": "1.3.0"}
    )

    assert "1.3.0" in widget.runtime_update_card.contentLabel.text()


def test_settings_runtime_refresh_reports_missing_native_helpers():
    QApplication.instance() or QApplication([])
    with patch.object(settings_module, "inspect_runtime_health", return_value=_report()):
        widget = SettingsWidget(None)

    with (
        patch.object(
            settings_module,
            "inspect_runtime_health",
            return_value=_report(native_complete=False),
        ),
        patch.object(
            settings_module.MessageManager,
            "send_warning_message",
        ) as warning,
    ):
        widget.refresh_runtime_health()

    warning.assert_called_once()
    assert "_phase" in warning.call_args.args[0]
