# Language Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add restart-applied Chinese / English language switching to NepTrainKit using Qt's official translation system.

**Architecture:** Add a focused `NepTrainKit.i18n` module that resolves the configured UI language, loads a packaged Qt `.qm` file, and installs a long-lived `QTranslator` during app setup. Settings owns the user-facing language selector and persists `ui.language`; individual UI modules mark visible strings with `self.tr(...)` or `QCoreApplication.translate(...)`. English remains the source/default language and Chinese is supplied by `neptrainkit_zh_CN.ts/.qm`.

**Tech Stack:** PySide6 `QTranslator`, `QLocale`, `QCoreApplication.translate`; qfluentwidgets Settings UI; existing SQLite-backed `Config`; pytest / pytest-qt; setuptools package data.

## Global Constraints

- First version is restart-applied only; do not implement live window retranslation.
- Store language config as `section = ui`, `option = language`, `value = auto | en_US | zh_CN`.
- UI display labels are `Auto`, `English`, and `中文`; persisted values remain `auto`, `en_US`, and `zh_CN`.
- Resolve `auto` to `zh_CN` when the system locale starts with `zh`; otherwise resolve to `en_US`.
- English is the source/default language; do not maintain `neptrainkit_en_US.qm`.
- Keep required parameters and data semantics in English: `Config_type`, `energy`, `force`, `virial`, `NEP`, `NEP89`, `DeepMD`, `VASP`, `extxyz`, `SpinTilt(...)`, `GSFE(...)`, `Bain(...)`, and config values such as `vispy`, `cpu`, `gpu`.
- Chinese text must be natural Chinese with preserved technical terms, spaces between Chinese and English/numbers, and full-width punctuation in Chinese sentences.
- Do not add `gettext`, custom dictionary translation systems, or new runtime dependencies.
- Do not touch unrelated dirty worktree files.

---

## File Structure

- Create `src/NepTrainKit/i18n.py`: language normalization, `auto` resolution, translation path lookup, `QTranslator` lifetime, and installation.
- Create `src/NepTrainKit/translations/neptrainkit_zh_CN.ts`: Qt Linguist XML catalog for the first translated UI slice.
- Create `src/NepTrainKit/translations/neptrainkit_zh_CN.qm`: compiled runtime translation catalog.
- Create `tools/update_translations.py`: repo-local helper that runs `pyside6-lupdate` and `pyside6-lrelease` with stable paths.
- Create `tests/test_i18n.py`: pure and light Qt tests for language resolution, fallback, and translator installation.
- Modify `src/NepTrainKit/main.py`: install translator during app configuration and mark main navigation strings translatable.
- Modify `src/NepTrainKit/ui/pages/settings.py`: add `Language` card, persist setting, and mark Settings text translatable.
- Modify `src/NepTrainKit/ui/messages.py`: mark default InfoBar titles and message-box titles translatable at emit/show time.
- Modify `src/NepTrainKit/ui/update.py`: mark high-frequency update messages translatable without translating `NEP89`.
- Modify `src/NepTrainKit/ui/pages/makedata.py`: mark high-frequency Make Data labels and messages translatable.
- Modify `src/NepTrainKit/ui/pages/show_nep.py`: mark high-frequency Show NEP labels and messages translatable.
- Modify `pyproject.toml`: include `translations/*.ts` and `translations/*.qm` as package data.

---

### Task 1: Core i18n Module

**Files:**
- Create: `src/NepTrainKit/i18n.py`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: `Config.get("ui", "language", "auto")`; `Config.set(...)` from later Settings task.
- Produces:
  - `SUPPORTED_LANGUAGES: tuple[str, ...]`
  - `LANGUAGE_LABELS: dict[str, str]`
  - `normalize_language(value: object | None) -> str`
  - `resolve_language(value: object | None = None, locale_name: str | None = None) -> str`
  - `translation_path(language: str) -> Path | None`
  - `install_translator(app: QApplication, language: object | None = None) -> str`
  - `current_language() -> str`

- [ ] **Step 1: Write failing tests for language normalization and auto resolution**

Create `tests/test_i18n.py` with these tests:

```python
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from NepTrainKit import i18n


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
    assert i18n.resolve_language("auto", "en_US") == "en_US"
    assert i18n.resolve_language("zh_CN", "en_US") == "zh_CN"
    assert i18n.resolve_language("en_US", "zh_CN") == "en_US"
    assert i18n.resolve_language("bad-value", "zh_CN") == "zh_CN"


def test_translation_path_only_for_chinese():
    path = i18n.translation_path("zh_CN")
    assert isinstance(path, Path)
    assert path.name == "neptrainkit_zh_CN.qm"
    assert i18n.translation_path("en_US") is None
    assert i18n.translation_path("auto") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_i18n.py
```

Expected: collection or import failure because `NepTrainKit.i18n` does not exist.

- [ ] **Step 3: Implement `src/NepTrainKit/i18n.py`**

Create `src/NepTrainKit/i18n.py`:

```python
"""Qt translation helpers for NepTrainKit."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PySide6.QtCore import QLocale, QTranslator
from PySide6.QtWidgets import QApplication
from loguru import logger

from NepTrainKit.config import Config

SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("auto", "en_US", "zh_CN")
LANGUAGE_LABELS: Final[dict[str, str]] = {
    "auto": "Auto",
    "en_US": "English",
    "zh_CN": "中文",
}
TRANSLATION_BASENAME: Final[str] = "neptrainkit"

_translator: QTranslator | None = None
_installed_language = "en_US"


def normalize_language(value: object | None) -> str:
    """Return a supported language config value, falling back to ``auto``."""
    text = str(value or "auto").strip()
    if text in SUPPORTED_LANGUAGES:
        return text
    return "auto"


def resolve_language(value: object | None = None, locale_name: str | None = None) -> str:
    """Resolve a configured language value to an actual runtime language."""
    language = normalize_language(value)
    if language != "auto":
        return language

    locale = locale_name if locale_name is not None else QLocale.system().name()
    if str(locale).lower().startswith("zh"):
        return "zh_CN"
    return "en_US"


def translation_path(language: str) -> Path | None:
    """Return the packaged ``.qm`` path for ``language`` when one is needed."""
    resolved = resolve_language(language, "en_US") if language == "auto" else normalize_language(language)
    if resolved != "zh_CN":
        return None
    return Path(__file__).resolve().parent / "translations" / f"{TRANSLATION_BASENAME}_{resolved}.qm"


def install_translator(app: QApplication, language: object | None = None) -> str:
    """Install the configured Qt translator and return the resolved language."""
    global _translator, _installed_language

    configured = normalize_language(language if language is not None else Config.get("ui", "language", "auto"))
    resolved = resolve_language(configured)
    _installed_language = resolved

    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None

    path = translation_path(resolved)
    if path is None:
        return resolved

    translator = QTranslator(app)
    if not path.exists():
        logger.warning("Translation file not found: {}", path)
        return resolved

    if not translator.load(str(path)):
        logger.warning("Failed to load translation file: {}", path)
        return resolved

    app.installTranslator(translator)
    _translator = translator
    return resolved


def current_language() -> str:
    """Return the most recently resolved runtime language."""
    return _installed_language
```

- [ ] **Step 4: Run Task 1 tests**

Run:

```bash
pytest -q tests/test_i18n.py
```

Expected: all 3 tests pass.

- [ ] **Step 5: Add translator-install fallback test**

Append to `tests/test_i18n.py`:

```python
def test_install_translator_falls_back_when_qm_is_missing(qtbot):
    app = QApplication.instance() or QApplication([])
    resolved = i18n.install_translator(app, "zh_CN")
    assert resolved == "zh_CN"
    assert i18n.current_language() == "zh_CN"
```

- [ ] **Step 6: Run Task 1 tests again**

Run:

```bash
pytest -q tests/test_i18n.py
```

Expected: all 4 tests pass, even before the `.qm` file exists.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add src/NepTrainKit/i18n.py tests/test_i18n.py
git commit -m "feat: add language resolution helpers"
```

Expected: commit succeeds with only the two Task 1 files staged.

---

### Task 2: App Startup and Package Data

**Files:**
- Modify: `src/NepTrainKit/main.py`
- Modify: `pyproject.toml`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: `install_translator(app: QApplication, language: object | None = None) -> str`.
- Produces: app startup installs translation before the main window is created; package data includes translation resources.

- [ ] **Step 1: Add failing test for package-data declaration**

Append to `tests/test_i18n.py`:

```python
import tomllib


def test_pyproject_includes_translation_package_data():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "NepTrainKit" in package_data
    assert "translations/*.qm" in package_data["NepTrainKit"]
    assert "translations/*.ts" in package_data["NepTrainKit"]
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
pytest -q tests/test_i18n.py::test_pyproject_includes_translation_package_data
```

Expected: FAIL because `NepTrainKit` package data is not declared yet.

- [ ] **Step 3: Update `pyproject.toml` package data**

Modify `[tool.setuptools.package-data]` in `pyproject.toml` to include the new `NepTrainKit` key:

```toml
[tool.setuptools.package-data]
"NepTrainKit" = ["translations/*.qm", "translations/*.ts"]
"NepTrainKit.Config" = ["config.sqlite","ptable.json","nep.json","nep89.txt" ]
```

- [ ] **Step 4: Run package-data test**

Run:

```bash
pytest -q tests/test_i18n.py::test_pyproject_includes_translation_package_data
```

Expected: PASS.

- [ ] **Step 5: Install translator during app configuration**

In `src/NepTrainKit/main.py`, add the import near existing NepTrainKit imports:

```python
from NepTrainKit.i18n import install_translator
```

Then modify `configure_app(app: QApplication) -> None` so translation is installed before windows/pages are created:

```python
def configure_app(app: QApplication) -> None:
    """Apply the same theme, font, stylesheet, and translator used by the desktop app."""
    set_light_theme(app)
    app.setApplicationName("NepTrainKit")
    install_translator(app)
    icon = _application_icon()
    app.setWindowIcon(icon)
    _set_macos_dock_icon(app, icon)
    font = QFont("Arial", 12)
    app.setFont(font)

    theme_file = QFile(":/theme/src/qss/theme.qss")
    if theme_file.open(QFile.OpenModeFlag.ReadOnly):
        theme = theme_file.readAll().data().decode("utf-8")  # pyright: ignore[reportArgumentType]
        theme_file.close()
        app.setStyleSheet(theme)
```

- [ ] **Step 6: Mark main navigation strings translatable**

In `src/NepTrainKit/main.py`, update `init_navigation` labels:

```python
self.addSubInterface(
    self.show_nep_interface,
    QIcon(':/images/src/images/show_nep.svg'),
    self.tr('NEP Dataset Display'),
)
self.addSubInterface(
    self.make_data_interface,
    QIcon(':/images/src/images/make.svg'),
    self.tr('Make Data'),
)
self.addSubInterface(
    self.data_manager_interface,
    QIcon(':/images/src/images/dataset.svg'),
    self.tr('Data Management'),
)
self.addSubInterface(
    self.setting_interface,
    FluentIcon.SETTING,
    self.tr('Settings'),
    NavigationItemPosition.BOTTOM,
)
```

- [ ] **Step 7: Run focused tests and import check**

Run:

```bash
pytest -q tests/test_i18n.py
python - <<'PY'
from NepTrainKit.main import create_app
app = create_app([])
print(app.applicationName())
PY
```

Expected: pytest passes and the Python snippet prints `NepTrainKit`.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add pyproject.toml src/NepTrainKit/main.py tests/test_i18n.py
git commit -m "feat: load translations at startup"
```

Expected: commit succeeds with only Task 2 files staged.

---

### Task 3: Settings Language Selector

**Files:**
- Modify: `src/NepTrainKit/ui/pages/settings.py`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: `LANGUAGE_LABELS`, `SUPPORTED_LANGUAGES`, `normalize_language` from `NepTrainKit.i18n`.
- Produces: `SettingsWidget.language_combo` with `userData` values `auto`, `en_US`, `zh_CN`; `_on_language_changed(index: int) -> None`.

- [ ] **Step 1: Add failing Settings test**

Append to `tests/test_i18n.py`:

```python
def test_settings_language_combo_persists_config(qtbot, monkeypatch):
    from NepTrainKit.config import Config
    from NepTrainKit.ui.pages.settings import SettingsWidget

    Config.set("ui", "language", "auto")
    widget = SettingsWidget(None)
    qtbot.addWidget(widget)

    values = [
        widget.language_combo.itemData(i)
        for i in range(widget.language_combo.count())
    ]
    assert values == ["auto", "en_US", "zh_CN"]

    index = values.index("zh_CN")
    widget.language_combo.setCurrentIndex(index)
    assert Config.get("ui", "language") == "zh_CN"
```

- [ ] **Step 2: Run the Settings test to verify it fails**

Run:

```bash
pytest -q tests/test_i18n.py::test_settings_language_combo_persists_config
```

Expected: FAIL because `SettingsWidget.language_combo` does not exist.

- [ ] **Step 3: Add imports to Settings page**

In `src/NepTrainKit/ui/pages/settings.py`, extend qfluentwidgets imports:

```python
from qfluentwidgets import SettingCardGroup, HyperlinkCard, PrimaryPushSettingCard, ExpandLayout, OptionsConfigItem, \
    OptionsValidator, EnumSerializer, SwitchSettingCard, FluentIcon, ScrollArea, SettingCard, ComboBox
```

Add message and i18n imports:

```python
from NepTrainKit.i18n import LANGUAGE_LABELS, SUPPORTED_LANGUAGES, normalize_language
from NepTrainKit.ui.messages import MessageManager
```

- [ ] **Step 4: Create the language card in `SettingsWidget.__init__`**

After `self.canvas_card` is created and before `auto_load_config`, add:

```python
language_config = normalize_language(Config.get("ui", "language", "auto"))
self.language_card = SettingCard(
    FluentIcon.SETTING,
    self.tr("Language"),
    self.tr("Restart NepTrainKit to apply language changes"),
    self.personal_group,
)
self.language_combo = ComboBox(self.language_card)
for value in SUPPORTED_LANGUAGES:
    self.language_combo.addItem(self.tr(LANGUAGE_LABELS[value]), userData=value)
self.language_combo.setCurrentIndex(SUPPORTED_LANGUAGES.index(language_config))
self.language_card.hBoxLayout.addWidget(self.language_combo, 0, Qt.AlignmentFlag.AlignRight)
self.language_card.hBoxLayout.addSpacing(16)
```

- [ ] **Step 5: Add the card to layout**

In `init_layout`, after `self.personal_group.addSettingCard(self.canvas_card)`, add:

```python
self.personal_group.addSettingCard(self.language_card)
```

- [ ] **Step 6: Connect the signal and persist changes**

In `init_signal`, add:

```python
self.language_combo.currentIndexChanged.connect(self._on_language_changed)
```

Add this method to `SettingsWidget`:

```python
def _on_language_changed(self, index: int) -> None:
    """Persist the selected UI language."""
    value = self.language_combo.itemData(index)
    value = normalize_language(value)
    Config.set("ui", "language", value)
    MessageManager.send_info_message(
        self.tr("Language saved. Restart NepTrainKit to apply it."),
        title=self.tr("Tip"),
    )
```

- [ ] **Step 7: Run the Settings test**

Run:

```bash
pytest -q tests/test_i18n.py::test_settings_language_combo_persists_config
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add src/NepTrainKit/ui/pages/settings.py tests/test_i18n.py
git commit -m "feat: add language setting"
```

Expected: commit succeeds with only Task 3 files staged.

---

### Task 4: Translate Settings and Core Messages

**Files:**
- Modify: `src/NepTrainKit/ui/pages/settings.py`
- Modify: `src/NepTrainKit/ui/messages.py`
- Modify: `src/NepTrainKit/ui/update.py`
- Create: `src/NepTrainKit/translations/neptrainkit_zh_CN.ts`
- Create: `src/NepTrainKit/translations/neptrainkit_zh_CN.qm`
- Create: `tools/update_translations.py`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: Qt translation markers in source code.
- Produces: first usable Chinese `.qm`; `tools/update_translations.py` CLI with `--no-lupdate` and `--no-lrelease` flags.

- [ ] **Step 1: Mark Settings page user-facing text with `self.tr(...)`**

In `src/NepTrainKit/ui/pages/settings.py`, convert Settings group names and card strings. Preserve option/config values such as enum values and `Config_type`. The first patch must cover these exact examples:

```python
self.personal_group = SettingCardGroup(self.tr('Personalization'), self.scrollWidget)
self.nep_group = SettingCardGroup(self.tr('NEP Settings'), self.scrollWidget)
self.plot_group = SettingCardGroup(self.tr('Plot Settings'), self.scrollWidget)
```

```python
self.optimization_forces_card = MyComboBoxSettingCard(
    OptionsConfigItem("forces", "forces", force_mode, OptionsValidator(ForcesMode), EnumSerializer(ForcesMode)),
    FluentIcon.BRUSH,
    self.tr('Force data format'),
    self.tr("Streamline data and speed up drawing"),
    texts=[mode.value for mode in ForcesMode],
    default=default_forces,
    parent=self.personal_group
)
```

```python
self.canvas_card = MyComboBoxSettingCard(
    OptionsConfigItem("canvas", "canvas", CanvasMode(canvas_type), OptionsValidator(CanvasMode), EnumSerializer(CanvasMode)),
    FluentIcon.BRUSH,
    self.tr('Canvas Engine'),
    self.tr("Choose GPU with vispy"),
    texts=[mode.value for mode in CanvasMode],
    default=canvas_type,
    parent=self.personal_group
)
```

Apply the same pattern to these Settings strings:

```text
Auto loading
Detect startup path data and load
Sort atoms
Sort atoms in structures when processing cards
Use card group menu
Group cards by "group" in console menu
Keep DeepMD subfolders
Preserve imported folder hierarchy when exporting deepmd/npy
Cache output files
Cache *.out and descriptor.out for faster reload (NEP & DeepMD)
Export significant digits
Significant digits for per-atom XYZ/extxyz values
Default Config_type
Tag assigned when source has no Config_type
Covalent radius coefficient
Coefficient used to detect bond length
NEP Backend
Select CPU/GPU or Auto detection
Data Precision
Choose storage precision for imported DFT/structure data
GPU Batch Size
Batch of frames processed GPU slice
Scatter edge color
Default edge color for points
Scatter face color
Default fill color for points
Face alpha (0-255)
Alpha channel for fill color
PyQtGraph scatter size
Marker size for PyQtGraph canvas
VisPy scatter size
Marker size for VisPy canvas
VisPy antialias
Marker antialias value for VisPy (0-2)
Structure background
Background color for lattice/structure viewer
Lattice line color
Line color for lattice edges in structure viewer
Selected color
Color for selected points
Show color
Color for highlighted "show" points
Current marker color
Color for current star marker
Current marker size
Size of current star marker
About
Open Help Page
Help
Discover new features and learn useful tips about NepTrainKit.
Submit Feedback
Help us improve NepTrainKit by providing feedback.
Check for Updates
Check and update
About NEP89
NEP official NEP89 large model
Version
New version available: v{pending_version}
```

- [ ] **Step 2: Mark message defaults translatable**

In `src/NepTrainKit/ui/messages.py`, import `QCoreApplication`:

```python
from PySide6.QtCore import QObject, Signal, Qt, QCoreApplication
```

Add helper:

```python
def _tr(text: str) -> str:
    return QCoreApplication.translate("MessageManager", text)
```

Update default titles:

```python
def send_info_message(cls, message, title=None):
    title = _tr("Tip") if title is None else title
```

Apply the same pattern:

```python
def send_success_message(cls, message, title=None):
    title = _tr("Success") if title is None else title

def send_warning_message(cls, message, title=None):
    title = _tr("Warning") if title is None else title

def send_error_message(cls, message, title=None):
    title = _tr("Error") if title is None else title

def send_message_box(cls, message, title=None):
    title = _tr("Tip") if title is None else title
```

- [ ] **Step 3: Mark high-frequency update strings translatable**

In `src/NepTrainKit/ui/update.py`, import `QCoreApplication` if not already present:

```python
from PySide6.QtCore import QCoreApplication
```

Add local helper near imports:

```python
def _tr(text: str) -> str:
    return QCoreApplication.translate("Update", text)
```

Wrap these exact static strings with `_tr(...)`, keeping interpolated versions as f-strings around translated prefixes:

```text
Update Check Failed
You are already using the latest version!
Open Releases
Close
Checking for updates, please wait...
Update available
Update large model completed!
No NEP89 release directory found in upstream repository.
Update
Cancel
```

For formatted text, use:

```python
MessageManager.send_info_message(_tr("Checking for updates, please wait..."))
```

Do not translate `NEP89`.

- [ ] **Step 4: Create translation maintenance script**

Create `tools/update_translations.py`:

```python
#!/usr/bin/env python
"""Update Qt translation sources and compiled catalogs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "NepTrainKit"
TS = SRC / "translations" / "neptrainkit_zh_CN.ts"


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise SystemExit(f"{name} not found. Install PySide6 tools in the active environment.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update NepTrainKit Qt translation files.")
    parser.add_argument("--no-lupdate", action="store_true", help="Skip updating the .ts source catalog.")
    parser.add_argument("--no-lrelease", action="store_true", help="Skip compiling the .qm runtime catalog.")
    args = parser.parse_args(argv)

    TS.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_lupdate:
        subprocess.run([_tool("pyside6-lupdate"), str(SRC), "-ts", str(TS)], check=True)

    if not args.no_lrelease:
        subprocess.run([_tool("pyside6-lrelease"), str(TS)], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 5: Generate `.ts`**

Run:

```bash
python tools/update_translations.py --no-lrelease
```

Expected: `src/NepTrainKit/translations/neptrainkit_zh_CN.ts` is created or updated.

- [ ] **Step 6: Fill Chinese translations for the first slice**

Edit `src/NepTrainKit/translations/neptrainkit_zh_CN.ts` so these source texts have these translations:

```text
Settings => 设置
NEP Dataset Display => NEP 数据集查看
Make Data => 构建数据
Data Management => 数据管理
Personalization => 个性化
NEP Settings => NEP 设置
Plot Settings => 绘图设置
Language => 语言
Restart NepTrainKit to apply language changes => 重启 NepTrainKit 后应用语言设置
Language saved. Restart NepTrainKit to apply it. => 语言设置已保存，重启 NepTrainKit 后生效。
Force data format => 力数据格式
Streamline data and speed up drawing => 精简数据并提升绘图速度
Canvas Engine => 绘图后端
Choose GPU with vispy => 使用 VisPy 时可选择 GPU 绘图
Auto loading => 自动加载
Detect startup path data and load => 启动时检测路径数据并加载
Sort atoms => 原子排序
Sort atoms in structures when processing cards => 处理卡片时对结构中的原子排序
Use card group menu => 使用卡片分组菜单
Group cards by "group" in console menu => 在控制台菜单中按 group 字段分组卡片
Keep DeepMD subfolders => 保留 DeepMD 子目录
Preserve imported folder hierarchy when exporting deepmd/npy => 导出 deepmd/npy 时保留导入目录层级
Cache output files => 缓存输出文件
Cache *.out and descriptor.out for faster reload (NEP & DeepMD) => 缓存 *.out 和 descriptor.out，加快 NEP 与 DeepMD 重新加载
Export significant digits => 导出有效数字
Significant digits for per-atom XYZ/extxyz values => XYZ/extxyz 逐原子数值的有效数字
Default Config_type => 默认 Config_type
Tag assigned when source has no Config_type => 源数据缺少 Config_type 时使用的标签
Covalent radius coefficient => 共价半径系数
Coefficient used to detect bond length => 用于判断键长的系数
NEP Backend => NEP 后端
Select CPU/GPU or Auto detection => 选择 CPU、GPU 或自动检测
Data Precision => 数据精度
Choose storage precision for imported DFT/structure data => 选择导入 DFT/结构数据的存储精度
GPU Batch Size => GPU 批大小
Batch of frames processed GPU slice => GPU 每次处理的结构帧数
About => 关于
Help => 帮助
Submit Feedback => 提交反馈
Check for Updates => 检查更新
Tip => 提示
Success => 成功
Warning => 警告
Error => 错误
Checking for updates, please wait... => 正在检查更新，请稍候……
You are already using the latest version! => 当前已是最新版本！
Update available => 发现新版本
Open Releases => 打开 Releases
Close => 关闭
Update => 更新
Cancel => 取消
```

- [ ] **Step 7: Compile `.qm`**

Run:

```bash
python tools/update_translations.py --no-lupdate
```

Expected: `src/NepTrainKit/translations/neptrainkit_zh_CN.qm` exists.

- [ ] **Step 8: Add translation smoke test**

Append to `tests/test_i18n.py`:

```python
from PySide6.QtCore import QCoreApplication


def test_chinese_qm_translates_core_label(qtbot):
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")
    translated = QCoreApplication.translate("NepTrainKitMainWindow", "Settings")
    assert translated in {"设置", "Settings"}
    assert i18n.current_language() == "zh_CN"
```

This accepts English if the local Qt catalog could not load, while still checking the install path does not crash. Manual verification in Task 6 checks visible Chinese.

- [ ] **Step 9: Run Task 4 tests**

Run:

```bash
pytest -q tests/test_i18n.py
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 4**

Run:

```bash
git add src/NepTrainKit/ui/pages/settings.py src/NepTrainKit/ui/messages.py src/NepTrainKit/ui/update.py src/NepTrainKit/translations/neptrainkit_zh_CN.ts src/NepTrainKit/translations/neptrainkit_zh_CN.qm tools/update_translations.py tests/test_i18n.py
git commit -m "feat: add Chinese UI translations"
```

Expected: commit succeeds with only Task 4 files staged.

---

### Task 5: Translate High-Frequency Show NEP and Make Data Text

**Files:**
- Modify: `src/NepTrainKit/ui/pages/makedata.py`
- Modify: `src/NepTrainKit/ui/pages/show_nep.py`
- Modify: `src/NepTrainKit/translations/neptrainkit_zh_CN.ts`
- Modify: `src/NepTrainKit/translations/neptrainkit_zh_CN.qm`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: existing `MessageManager`, Qt translation catalog.
- Produces: first-pass Chinese coverage for primary workflow labels and messages.

- [ ] **Step 1: Mark Make Data high-frequency text**

In `src/NepTrainKit/ui/pages/makedata.py`, convert these exact strings to `self.tr(...)` when inside `MakeDataWidget` methods:

```text
Only .xyz .vasp .cif or json files are supported for import.
Folder for Custom Cards
success load {len(structures_list)} structures.
Success load {len(structures_list)} structures.
Cards are still running. Please wait for the current run to finish.
No card selected. Please select a card in the workspace.
Perturbation training set created successfully.
no card
No cards in workspace.
Card configuration exported successfully.
Card configuration JSON copied to clipboard.
Clipboard does not contain card JSON.
Invalid card configuration file: {exc}
Failed to load {name}: {exc}
Added {added_count} card configuration(s).
```

For f-strings, use:

```python
MessageManager.send_success_message(
    self.tr("success load {count} structures.").format(count=len(structures_list))
)
self.dataset_info_label.setText(
    self.tr("Success load {count} structures.").format(count=len(structures_list))
)
```

Use `{count}`, `{name}`, and `{error}` placeholders instead of embedding Python expressions directly in source strings.

- [ ] **Step 2: Mark Show NEP high-frequency text**

In `src/NepTrainKit/ui/pages/show_nep.py`, convert these exact strings to `self.tr(...)` in `ShowNepWidget` methods:

```text
Export Selected ({selected})…
Export Removed ({removed})…
Export Active ({active})…
Current structure (original file index):
Searching…
Indexing…
Current file: {file_name}
NEP data has not been loaded yet!
No active structures to export.
Please select some structures first!
No removed structures to export.
File exported to: {save_file_path}
Failed to build search completer cache: {msg}
Search failed: {msg}
Arrow overlay is unavailable for current structure canvas backend.
No vector data available
No bad structures tagged.
Failed to delete rejected structures.
The distance between atoms is too small, and the structure may be unreasonable.
Please enter a search query.
unsupported file format
Failed to switch NEP model
```

For dynamic values, use `.format(...)`:

```python
self.path_label.setText(self.tr("Current file: {file_name}").format(file_name=file_name))
```

Keep data labels such as `force`, `energy`, `virial`, and `Config_type` unchanged when they describe fields or exported data.

- [ ] **Step 3: Update translation catalog**

Run:

```bash
python tools/update_translations.py --no-lrelease
```

Expected: `.ts` includes the new `MakeDataWidget` and `ShowNepWidget` messages.

- [ ] **Step 4: Fill Chinese translations for workflow messages**

Edit `src/NepTrainKit/translations/neptrainkit_zh_CN.ts` with these translations:

```text
Only .xyz .vasp .cif or json files are supported for import. => 仅支持导入 .xyz、.vasp、.cif 或 json 文件。
Folder for Custom Cards => 自定义卡片目录
success load {count} structures. => 已成功加载 {count} 个结构。
Success load {count} structures. => 已成功加载 {count} 个结构。
Cards are still running. Please wait for the current run to finish. => 卡片仍在运行，请等待当前任务完成。
No card selected. Please select a card in the workspace. => 尚未选择卡片，请先在工作区选择一个卡片。
Perturbation training set created successfully. => 扰动训练集已创建。
no card => 没有卡片
No cards in workspace. => 工作区中没有卡片。
Card configuration exported successfully. => 卡片配置已导出。
Card configuration JSON copied to clipboard. => 卡片配置 JSON 已复制到剪贴板。
Clipboard does not contain card JSON. => 剪贴板中没有卡片 JSON。
Invalid card configuration file: {error} => 卡片配置文件无效：{error}
Failed to load {name}: {error} => 加载 {name} 失败：{error}
Added {count} card configuration(s). => 已添加 {count} 个卡片配置。
Export Selected ({selected})… => 导出已选择结构（{selected}）…
Export Removed ({removed})… => 导出已移除结构（{removed}）…
Export Active ({active})… => 导出当前保留结构（{active}）…
Current structure (original file index): => 当前结构（原始文件序号）：
Searching… => 正在搜索…
Indexing… => 正在建立索引…
Current file: {file_name} => 当前文件：{file_name}
NEP data has not been loaded yet! => 尚未加载 NEP 数据！
No active structures to export. => 没有可导出的保留结构。
Please select some structures first! => 请先选择一些结构！
No removed structures to export. => 没有可导出的已移除结构。
File exported to: {save_file_path} => 文件已导出到：{save_file_path}
Failed to build search completer cache: {msg} => 构建搜索补全缓存失败：{msg}
Search failed: {msg} => 搜索失败：{msg}
Arrow overlay is unavailable for current structure canvas backend. => 当前结构绘图后端不支持箭头叠加。
No vector data available => 没有可用的矢量数据
No bad structures tagged. => 没有标记为异常的结构。
Failed to delete rejected structures. => 删除已剔除结构失败。
The distance between atoms is too small, and the structure may be unreasonable. => 原子间距过小，结构可能不合理。
Please enter a search query. => 请输入搜索内容。
unsupported file format => 不支持的文件格式
Failed to switch NEP model => 切换 NEP 模型失败
```

- [ ] **Step 5: Compile `.qm`**

Run:

```bash
python tools/update_translations.py --no-lupdate
```

Expected: `.qm` is regenerated successfully.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest -q tests/test_i18n.py
git diff --check
```

Expected: pytest passes and `git diff --check` prints no errors.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add src/NepTrainKit/ui/pages/makedata.py src/NepTrainKit/ui/pages/show_nep.py src/NepTrainKit/translations/neptrainkit_zh_CN.ts src/NepTrainKit/translations/neptrainkit_zh_CN.qm tests/test_i18n.py
git commit -m "feat: translate primary workflow messages"
```

Expected: commit succeeds with only Task 5 files staged.

---

### Task 6: Final Verification

**Files:**
- Verify: all files touched by Tasks 1-5

**Interfaces:**
- Consumes: completed implementation.
- Produces: verified working state, no code changes unless a verification failure reveals a specific bug.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest -q tests/test_i18n.py
```

Expected: all tests pass.

- [ ] **Step 2: Run existing lightweight Settings-adjacent tests**

Run:

```bash
pytest -q tests/test_update_notifier.py tests/test_threads.py
```

Expected: tests pass. If one fails due to environment/network assumptions unrelated to i18n, capture the failure text and do not mask it.

- [ ] **Step 3: Run translation maintenance script**

Run:

```bash
python tools/update_translations.py
```

Expected: command exits 0 and regenerates `.ts/.qm`.

- [ ] **Step 4: Check for accidental translation of protected terms**

Run:

```bash
python - <<'PY'
from pathlib import Path
ts = Path("src/NepTrainKit/translations/neptrainkit_zh_CN.ts").read_text(encoding="utf-8")
protected = ["Config_type", "NEP89", "DeepMD", "VASP", "extxyz", "SpinTilt", "GSFE", "Bain"]
missing = [term for term in protected if term not in ts]
print("protected terms present:", ", ".join(term for term in protected if term in ts))
if missing:
    print("not present in catalog:", ", ".join(missing))
PY
```

Expected: command prints a list and exits 0. Absence from the catalog is acceptable when a protected term was never part of a translated source string.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Manual smoke check English mode**

Run:

```bash
python - <<'PY'
from NepTrainKit.config import Config
Config.set("ui", "language", "en_US")
print(Config.get("ui", "language"))
PY
python -m NepTrainKit.main
```

Expected: the app opens in English; Settings shows `Language`; close the app manually after checking.

- [ ] **Step 7: Manual smoke check Chinese mode**

Run:

```bash
python - <<'PY'
from NepTrainKit.config import Config
Config.set("ui", "language", "zh_CN")
print(Config.get("ui", "language"))
PY
python -m NepTrainKit.main
```

Expected: the app opens; migrated navigation and Settings text show Chinese; protected terms such as `NEP`, `DeepMD`, `VASP`, and `Config_type` remain English; close the app manually after checking.

- [ ] **Step 8: Restore Auto mode**

Run:

```bash
python - <<'PY'
from NepTrainKit.config import Config
Config.set("ui", "language", "auto")
print(Config.get("ui", "language"))
PY
```

Expected: prints `auto`.

- [ ] **Step 9: Confirm no uncommitted implementation changes remain**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing local artifacts or user changes are listed. If Task 6 revealed an i18n bug, return to the task that introduced it, make a focused fix there, rerun Task 6, and commit that fix with the files from that task only.
