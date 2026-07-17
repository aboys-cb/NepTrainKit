from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication
import pytest

from NepTrainKit import i18n
from NepTrainKit.ui import update

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "update_translations.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("update_translations", _SCRIPT_PATH)
assert _SCRIPT_SPEC and _SCRIPT_SPEC.loader
update_translations = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules.setdefault("update_translations", update_translations)
_SCRIPT_SPEC.loader.exec_module(update_translations)


@pytest.fixture(autouse=True)
def _restore_application_language_after_test():
    """Keep translator state from leaking into unrelated Qt test modules."""
    app = QApplication.instance()
    if app is not None:
        i18n.install_translator(app, "en_US")
    yield
    app = QApplication.instance()
    if app is not None:
        i18n.install_translator(app, "en_US")

_MAKE_DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "NepTrainKit" / "ui" / "pages" / "makedata.py"
_SHOW_NEP_PATH = Path(__file__).resolve().parents[1] / "src" / "NepTrainKit" / "ui" / "pages" / "show_nep.py"
_UPDATE_PATH = Path(__file__).resolve().parents[1] / "src" / "NepTrainKit" / "ui" / "update.py"

_TASK5_MAKE_DATA_MARKERS = (
    'self.tr("Only .xyz .vasp .cif or json files are supported for import.")',
    'self.tr("Folder for Custom Cards")',
    'self.tr("success load {count} structures.")',
    'self.tr("Success load {count} structures.")',
    'self.tr("Cards are still running. Please wait for the current run to finish.")',
    'self.tr("Please import the structure file first. You can drag it in directly or import it from the upper left corner!")',
    'self.tr("No card selected. Please select a card in the workspace.")',
    'self.tr("Perturbation training set created successfully.")',
    'self.tr("no card")',
    'self.tr("No cards in workspace.")',
    'self.tr("Card configuration exported successfully.")',
    'self.tr("Card configuration JSON copied to clipboard.")',
    'self.tr("Clipboard does not contain card JSON.")',
    'self.tr("Invalid card configuration file: {error}")',
    'self.tr("Failed to load {name}: {error}")',
    'self.tr("Added {count} card configuration(s).")',
)

_TASK5_SHOW_NEP_MARKERS = (
    'self.tr("Open File…")',
    'self.tr("Open Folder…")',
    'self.tr("Export All…")',
    'self.tr("Export Selected ({selected})…")',
    'self.tr("Export Removed ({removed})…")',
    'self.tr("Export Active ({active})…")',
    'self.tr("Current structure (original file index):")',
    'self.tr("Searching…")',
    'self.tr("Indexing…")',
    'self.tr("Current file: {file_name}")',
    'self.tr("NEP data has not been loaded yet!")',
    'self.tr("No active structures to export.")',
    'self.tr("Please select some structures first!")',
    'self.tr("No removed structures to export.")',
    'self.tr("File exported to: {save_file_path}")',
    'self.tr("Failed to build search completer cache: {msg}")',
    'self.tr("Search failed: {msg}")',
    'self.tr("Arrow overlay is unavailable for current structure canvas backend.")',
    'self.tr("No vector data available")',
    'self.tr("No bad structures tagged.")',
    'self.tr("Failed to delete rejected structures.")',
    'self.tr("Confirm")',
    'self.tr("A working directory already exists. Loading a new directory will erase the previous results.\\nDo you want to load the new working path?")',
    'self.tr("This will delete {count} structures marked as bad.\\nDo you want to continue?")',
    'self.tr("The distance between atoms is too small, and the structure may be unreasonable.")',
    'self.tr("Please enter a search query.")',
    'self.tr("unsupported file format")',
    'self.tr("Failed to switch NEP model")',
)

_FINAL_REVIEW_MAKE_DATA_MARKERS = (
    'self.tr("Export Card Config")',
    'self.tr("Import Card Config")',
    'self.tr("Paste Card JSON")',
    'self.tr("Load structure failed: {path}")',
)

_FINAL_REVIEW_UPDATE_MARKERS = (
    '_tr("Upgrade command: {command}")',
    '_tr("Pip Upgrade")',
)

_TASK5_TRANSLATIONS = {
    "MakeDataWidget": {
        "Only .xyz .vasp .cif or json files are supported for import.": "仅支持导入 .xyz、.vasp、.cif 或 json 文件。",
        "Folder for Custom Cards": "自定义卡片目录",
        "success load {count} structures.": "已成功加载 {count} 个结构。",
        "Success load {count} structures.": "已成功加载 {count} 个结构。",
        "Cards are still running. Please wait for the current run to finish.": "卡片仍在运行，请等待当前任务完成。",
        "Please import the structure file first. You can drag it in directly or import it from the upper left corner!": "请先导入结构文件。你可以直接拖入，或从左上角导入。",
        "No card selected. Please select a card in the workspace.": "尚未选择卡片，请先在工作区选择一个卡片。",
        "Perturbation training set created successfully.": "扰动训练集已创建。",
        "no card": "没有卡片",
        "No cards in workspace.": "工作区中没有卡片。",
        "Card configuration exported successfully.": "卡片配置已导出。",
        "Card configuration JSON copied to clipboard.": "卡片配置 JSON 已复制到剪贴板。",
        "Clipboard does not contain card JSON.": "剪贴板中没有卡片 JSON。",
        "Invalid card configuration file: {error}": "卡片配置文件无效：{error}",
        "Failed to load {name}: {error}": "加载 {name} 失败：{error}",
        "Added {count} card configuration(s).": "已添加 {count} 个卡片配置。",
        "Export Card Config": "导出卡片配置",
        "Import Card Config": "导入卡片配置",
        "Paste Card JSON": "粘贴卡片 JSON",
        "Load structure failed: {path}": "加载结构失败：{path}",
    },
    "ShowNepWidget": {
        "Open File…": "打开文件…",
        "Open Folder…": "打开文件夹…",
        "Export All…": "导出全部…",
        "Export Selected ({selected})…": "导出已选择结构（{selected}）…",
        "Export Removed ({removed})…": "导出已移除结构（{removed}）…",
        "Export Active ({active})…": "导出当前保留结构（{active}）…",
        "Current structure (original file index):": "当前结构（原始文件序号）：",
        "Searching…": "正在搜索…",
        "Indexing…": "正在建立索引…",
        "Current file: {file_name}": "当前文件：{file_name}",
        "NEP data has not been loaded yet!": "尚未加载 NEP 数据！",
        "No active structures to export.": "没有可导出的保留结构。",
        "Please select some structures first!": "请先选择一些结构！",
        "No removed structures to export.": "没有可导出的已移除结构。",
        "File exported to: {save_file_path}": "文件已导出到：{save_file_path}",
        "Failed to build search completer cache: {msg}": "构建搜索补全缓存失败：{msg}",
        "Search failed: {msg}": "搜索失败：{msg}",
        "Arrow overlay is unavailable for current structure canvas backend.": "当前结构绘图后端不支持箭头叠加。",
        "No vector data available": "没有可用的矢量数据",
        "No bad structures tagged.": "没有标记为异常的结构。",
        "Failed to delete rejected structures.": "删除已剔除结构失败。",
        "Confirm": "确认",
        "A working directory already exists. Loading a new directory will erase the previous results.\nDo you want to load the new working path?": "已有工作目录。加载新目录会清除之前的结果。\n仍要加载新的工作路径吗？",
        "This will delete {count} structures marked as bad.\nDo you want to continue?": "这将删除 {count} 个标记为异常的结构。\n要继续吗？",
        "The distance between atoms is too small, and the structure may be unreasonable.": "原子间距过小，结构可能不合理。",
        "Please enter a search query.": "请输入搜索内容。",
        "unsupported file format": "不支持的文件格式",
        "Failed to switch NEP model": "切换 NEP 模型失败",
    },
    "Update": {
        "Upgrade command: {command}": "升级命令：{command}",
        "Pip Upgrade": "Pip 升级",
    },
}


def test_normalize_language_accepts_only_supported_values():
    assert i18n.normalize_language("auto") == "auto"
    assert i18n.normalize_language("en_US") == "en_US"
    assert i18n.normalize_language("zh_CN") == "zh_CN"
    assert i18n.normalize_language("  zh_CN  ") == "zh_CN"
    assert i18n.normalize_language("zh") == "auto"
    assert i18n.normalize_language(None) == "auto"


def test_resolve_language_from_locale_name():
    assert i18n.resolve_language("auto", "zh_CN") == "zh_CN"
    assert i18n.resolve_language("auto", "zh_Hans_CN") == "zh_CN"
    assert i18n.resolve_language("auto", "zh-Hans-CN") == "zh_CN"
    assert i18n.resolve_language("auto", "en_US") == "en_US"
    assert i18n.resolve_language("zh_CN", "en_US") == "zh_CN"
    assert i18n.resolve_language("en_US", "zh_CN") == "en_US"
    assert i18n.resolve_language("bad-value", "zh_CN") == "zh_CN"


def test_resolve_language_uses_system_candidates(monkeypatch):
    monkeypatch.setattr(i18n, "_system_locale_candidates", lambda: ["zh-Hans-CN", "en_US"])
    assert i18n.resolve_language("auto") == "zh_CN"


def test_resolve_language_uses_next_supported_candidate(monkeypatch):
    monkeypatch.setattr(i18n, "_system_locale_candidates", lambda: ["C", "POSIX", "en_US", "zh_CN"])
    assert i18n.resolve_language("auto") == "en_US"


def test_environment_locale_candidates_prefer_ui_message_language(monkeypatch):
    monkeypatch.setenv("LANGUAGE", "zh_CN:en_US")
    monkeypatch.setenv("LC_MESSAGES", "en_US.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    assert i18n._environment_locale_candidates()[:4] == [
        "zh_CN",
        "en_US",
        "en_US.UTF-8",
        "en_US.UTF-8",
    ]


def test_translation_path_only_for_chinese():
    path = i18n.translation_path("zh_CN")
    assert isinstance(path, Path)
    assert path.name == "neptrainkit_zh_CN.qm"
    assert i18n.translation_path("en_US") is None
    assert i18n.translation_path("auto") is None


def test_install_translator_falls_back_when_qm_is_missing(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(i18n, "translation_path", lambda language: tmp_path / f"{language}.qm")
    resolved = i18n.install_translator(app, "zh_CN")
    assert resolved == "zh_CN"
    assert i18n.current_language() == "zh_CN"
    assert QCoreApplication.translate("MessageManager", "Tip") == "Tip"


def test_pyproject_includes_translation_package_data():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "NepTrainKit" in package_data
    assert "translations/*.qm" in package_data["NepTrainKit"]
    assert "translations/*.ts" in package_data["NepTrainKit"]


def test_settings_language_combo_persists_config():
    from NepTrainKit.config import Config
    from NepTrainKit.ui.pages.settings import SettingsWidget

    QApplication.instance() or QApplication([])
    prev_language = Config.get("ui", "language")
    try:
        Config.set("ui", "language", "auto")
        widget = SettingsWidget(None)

        values = [
            widget.language_combo.itemData(i)
            for i in range(widget.language_combo.count())
        ]
        assert values == ["auto", "en_US", "zh_CN"]

        index = values.index("zh_CN")
        widget.language_combo.setCurrentIndex(index)
        assert Config.get("ui", "language") == "zh_CN"
    finally:
        if prev_language is None:
            Config.delete("ui", "language")
        else:
            Config.set("ui", "language", prev_language)


def test_chinese_qm_translates_core_label():
    app = QApplication.instance() or QApplication([])
    qm_path = i18n.translation_path("zh_CN")
    assert qm_path is not None
    assert qm_path.exists()

    i18n.install_translator(app, "zh_CN")
    translated = QCoreApplication.translate("NepTrainKitMainWindow", "Settings")
    assert translated == "设置"
    assert QCoreApplication.translate("SettingsWidget", "Log level") == "日志等级"
    assert i18n.current_language() == "zh_CN"


def test_core_message_defaults_are_translated_in_chinese_mode():
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")

    from NepTrainKit.core.message import MessageManager as CoreMessageManager
    from NepTrainKit.ui.messages import MessageManager as UiMessageManager

    message_titles: list[str] = []
    box_titles: list[str] = []

    class _Signal:
        def __init__(self, sink: list[str], title_index: int):
            self._sink = sink
            self._title_index = title_index

        def emit(self, *args):
            self._sink.append(args[self._title_index])

    class _FakeInstance:
        showMessageSignal = _Signal(message_titles, 2)
        showBoxSignal = _Signal(box_titles, 1)

    previous_instance = UiMessageManager._instance
    try:
        UiMessageManager._instance = _FakeInstance()
        CoreMessageManager.register_sink(UiMessageManager)

        CoreMessageManager.send_info_message("info")
        CoreMessageManager.send_success_message("success")
        CoreMessageManager.send_warning_message("warning")
        CoreMessageManager.send_error_message("error")
        CoreMessageManager.send_message_box("box")
    finally:
        UiMessageManager._instance = previous_instance
        CoreMessageManager.reset_sink()

    assert message_titles == ["提示", "成功", "警告", "错误"]
    assert box_titles == ["提示"]


def test_chinese_qm_translates_helper_contexts():
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")

    assert QCoreApplication.translate("MessageManager", "Tip") == "提示"
    assert QCoreApplication.translate("Update", "Update available") == "发现新版本"


def test_task5_source_strings_are_marked_for_translation():
    make_data_text = _MAKE_DATA_PATH.read_text(encoding="utf-8")
    show_nep_text = _SHOW_NEP_PATH.read_text(encoding="utf-8")
    update_text = _UPDATE_PATH.read_text(encoding="utf-8")

    for marker in _TASK5_MAKE_DATA_MARKERS:
        assert marker in make_data_text

    for marker in _TASK5_SHOW_NEP_MARKERS:
        assert marker in show_nep_text

    for marker in _FINAL_REVIEW_MAKE_DATA_MARKERS:
        assert marker in make_data_text

    for marker in _FINAL_REVIEW_UPDATE_MARKERS:
        assert marker in update_text


def test_task5_chinese_qm_translates_primary_workflow_strings():
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")

    for context, entries in _TASK5_TRANSLATIONS.items():
        for source, expected in entries.items():
            assert QCoreApplication.translate(context, source) == expected


def test_translation_script_discovers_standard_qt_markers(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    markers = {
        "uses_tr.py": "label = self.tr('Settings')\n",
        "uses_helper.py": "label = _tr('Update available')\n",
        "uses_translate.py": "label = QCoreApplication.translate('Update', 'Close')\n",
        "plain.py": "label = 'not translated'\n",
        "dialog.ui": "<ui></ui>\n",
    }
    for name, content in markers.items():
        (src / name).write_text(content, encoding="utf-8")

    monkeypatch.setattr(update_translations, "SRC", src)

    files = {Path(path).name for path in update_translations._source_files()}

    assert files == {"uses_tr.py", "uses_helper.py", "uses_translate.py", "dialog.ui"}


def test_translation_script_normalizes_helper_contexts_from_update_path(tmp_path):
    ts_path = tmp_path / "neptrainkit_zh_CN.ts"
    ts_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<TS version="2.1" language="zh_CN">
  <context>
    <name></name>
    <message>
      <location filename="src/NepTrainKit/ui/update.py" line="10" />
      <source>Close</source>
      <translation>关闭</translation>
    </message>
  </context>
</TS>
""",
        encoding="utf-8",
    )

    update_translations._normalize_helper_contexts(ts_path)
    text = ts_path.read_text(encoding="utf-8")

    assert "<name>Update</name>" in text
    assert 'filename="src/NepTrainKit/ui/update.py"' in text
    assert "<name />" not in text


def test_startup_pending_notice_is_localized(monkeypatch):
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")
    notifier = update.AutoUpdateNotifier()
    captured: dict[str, str] = {}

    monkeypatch.setattr(update, "get_pending_update_version", lambda: "1.2.3")
    monkeypatch.setattr(
        update.MessageManager,
        "send_info_message",
        lambda message, title=None: captured.update({"message": message, "title": title or ""}),
    )

    notifier._show_startup_pending_notice()

    assert captured["title"] == "发现新版本"
    assert captured["message"] == "发现新版本 v1.2.3。请前往 设置 > 关于 > 检查更新 查看详情。"


def test_auto_update_summary_notice_is_localized(monkeypatch):
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")

    result = {
        "ok": True,
        "has_update": True,
        "latest_version": "1.2.3",
        "notes": "Detailed release notes",
        "summary": "修复了若干问题。",
        "release_url": "https://example.com/release",
    }
    captured: dict[str, str] = {}
    notifier = update.AutoUpdateNotifier()

    monkeypatch.setattr(update.Config, "set", lambda *args, **kwargs: None)
    monkeypatch.setattr(update.Config, "get", lambda *args, **kwargs: "")
    monkeypatch.setattr(update, "_set_pending_update_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(update, "should_notify_version", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        update.MessageManager,
        "send_info_message",
        lambda message, title=None: captured.update({"message": message, "title": title or ""}),
    )

    notifier._handle_auto_result(result)

    assert captured["title"] == "发现新版本"
    assert captured["message"] == "发现新版本 v1.2.3。修复了若干问题。请前往 设置 > 关于 > 检查更新 查看详情。"


def test_nep89_prompt_and_warehouse_warning_are_localized(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")
    worker = update.UpdateNEP89Woker(None)
    warnings: list[tuple[str, str | None]] = []
    prompt: dict[str, str] = {}

    class FakeResponse:
        status_code = 503

    class FakeMessageBox:
        def __init__(self, title, content, parent):
            prompt["title"] = title
            prompt["content"] = content
            self.yesButton = type("Button", (), {"setText": lambda self, text: prompt.setdefault("yes", text)})()
            self.cancelButton = type("Button", (), {"setText": lambda self, text: prompt.setdefault("cancel", text)})()

        def exec_(self):
            return None

        def result(self):
            return 0

    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    nep_json = config_dir / "nep.json"
    nep_json.write_text('{"date": 20240101}', encoding="utf-8")
    monkeypatch.setattr(update, "module_path", tmp_path)
    monkeypatch.setattr(update, "MessageBox", FakeMessageBox)
    monkeypatch.setattr(
        update.MessageManager,
        "send_warning_message",
        lambda message, title=None: warnings.append((message, title)),
    )

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", type("Requests", (), {"get": staticmethod(fake_get)}))

    worker._check_update()
    worker._check_update_call_back(20250101)

    assert warnings == [("无法访问仓库目录，状态码：503", None)]
    assert prompt["title"] == "发现新版本"
    assert prompt["content"] == "检测到大模型有新版本：20250101"
    assert prompt["yes"] == "更新"
    assert prompt["cancel"] == "取消"
