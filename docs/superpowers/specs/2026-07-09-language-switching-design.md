# NepTrainKit 语言切换设计

## 背景

NepTrainKit 目前以英文界面为主，README 和项目说明已经逐步中文化。为了让中文用户的使用体验更连贯，同时保留英文界面对海外用户和必要技术参数的友好性，软件需要支持中文和英文切换。

本设计采用 Qt 官方国际化方案：源码中的英文文本作为默认文案，中文译文通过 Qt Linguist 的 `.ts` / `.qm` 文件提供。第一版只做“重启后生效”的语言切换，不做运行时动态刷新。

## 目标

- 在 Settings 页面提供语言选择：`Auto`、`English`、`中文`。
- 将语言配置持久化到现有 `Config` 存储中。
- 启动时按配置加载中文翻译文件；英文作为默认回退语言。
- 第一版覆盖主导航、Settings、About/Update、常见消息、`Show NEP` 和 `Make Data` 高频界面文案。
- 中文译文自然、清晰，遵守中文技术文档排版习惯：中英文之间加空格，中文语境使用全角标点，保留必要英文术语。

## 非目标

- 不做切换后立即刷新当前窗口。
- 不翻译日志、异常堆栈、开发者注释。
- 不翻译数据字段、模型字段、文件格式名和必要参数名。
- 不一次性迁移所有 Make Dataset 卡片参数说明。
- 不引入 `gettext` 或自维护字典式翻译系统。

## 语言配置

配置存储在现有 SQLite 配置系统中：

```text
section = ui
option = language
value = auto | en_US | zh_CN
```

显示文案和配置值分离：

| 设置页显示 | 配置值 |
| --- | --- |
| Auto | `auto` |
| English | `en_US` |
| 中文 | `zh_CN` |

`auto` 的解析规则：

- 系统 locale 以 `zh` 开头时使用 `zh_CN`。
- 其他情况使用 `en_US`。
- 配置值非法时回退 `auto`。

## 架构

新增 `src/NepTrainKit/i18n.py`，集中处理国际化逻辑：

- 读取并校验语言配置。
- 解析 `auto`。
- 计算翻译文件路径。
- 创建并持有 `QTranslator` 实例，避免被 Python 回收。
- 安装翻译器到 `QApplication`。
- 翻译文件缺失或加载失败时回退英文，并记录日志。

主入口在 `create_app()` 或 `configure_app()` 阶段调用该模块。窗口和页面只负责用 Qt 翻译 API 标记文案，不直接处理语言加载。

## 翻译文件

翻译资源放在包内：

```text
src/NepTrainKit/translations/
  neptrainkit_zh_CN.ts
  neptrainkit_zh_CN.qm
```

英文作为源码默认语言，不维护 `neptrainkit_en_US.qm`。这样能减少一份翻译资源，也符合 Qt 常见用法。

后续维护使用 Qt 工具：

```bash
pyside6-lupdate src/NepTrainKit -ts src/NepTrainKit/translations/neptrainkit_zh_CN.ts
pyside6-lrelease src/NepTrainKit/translations/neptrainkit_zh_CN.ts
```

可以新增一个轻量脚本或文档命令封装这两步，避免维护命令漂移。

## 代码迁移方式

`QObject` 子类中的用户可见文本使用：

```python
self.tr("Settings")
```

非 `QObject` 场景使用：

```python
QCoreApplication.translate("Context", "Settings")
```

迁移时只标记 UI 文案，不标记数据语义。以下内容保持英文：

- 数据字段：`Config_type`、`energy`、`force`、`virial`
- 模型和格式：`NEP`、`NEP89`、`DeepMD`、`VASP`、`extxyz`
- 配置值：`auto`、`en_US`、`zh_CN`、`vispy`、`cpu`、`gpu`
- 结构标签：`SpinTilt(...)`、`GSFE(...)`、`Bain(...)`

## 第一版覆盖范围

第一轮迁移这些用户高频可见文本：

- 主窗口导航：`NEP Dataset Display`、`Make Data`、`Data Management`、`Settings`
- Settings 页面：分组标题、设置项标题、说明、按钮、更新提示
- 全局消息：成功、失败、警告、导入导出、更新检查提示
- `Show NEP` 页面：常见按钮、标签、空状态、导入导出提示
- `Make Data` 页面：常见按钮、标签、工作区提示、卡片配置导入导出提示
- 项目内显式设置的常见按钮：`Ok`、`Cancel`、`Close`、`Update`

Make Dataset 卡片的参数说明后续按模块逐步迁移，不在第一版强行全量完成。

## Settings 交互

Settings 页面新增 `Language` 设置卡，放在 `Personalization` 分组内。

用户修改语言后：

1. 保存配置到 `Config.set("ui", "language", value)`。
2. 显示提示：语言设置将在重启 NepTrainKit 后生效。
3. 不尝试刷新当前窗口。

中文提示建议：

```text
语言设置已保存，重启 NepTrainKit 后生效。
```

英文提示建议：

```text
Language saved. Restart NepTrainKit to apply it.
```

## 中文译文原则

中文翻译遵守 `chinese-documentation` 规范：

- 中文表达优先自然，不按英文句式直译。
- 中文与英文、数字之间加空格。
- 中文语境使用全角标点。
- `NEP`、`DeepMD`、`VASP`、`Config_type` 等专业术语保留英文。
- 按用户理解选择译名，例如 `Canvas Engine` 可译为“绘图后端”，`NEP Backend` 译为“NEP 后端”。
- 避免机翻味，例如不使用“高速缓存输出文件”这类生硬说法。

## 错误处理

- 配置值非法：按 `auto` 处理。
- `zh_CN` 翻译文件不存在：英文启动，记录日志。
- `.qm` 加载失败：英文启动，记录日志。
- Qt 工具不可用：不影响运行，只影响维护翻译文件。

## 打包

需要确保 `.qm` 文件进入包数据。当前项目已有 package data 配置，需要补充 `translations/*.qm`，必要时也保留 `.ts` 供源码分发和维护。

打包后语言文件应通过包内路径加载，不能依赖当前工作目录。

## 测试

测试分三层：

- 单元测试：验证 `auto / zh_CN / en_US` 解析、非法配置回退、翻译文件缺失时回退英文。
- GUI 构造测试：创建 `QApplication`，确认 Settings 中存在语言设置项，修改后能写入 `Config`。
- 包数据检查：确认 `.qm` 在安装或源码运行时可被定位。

第一版不要求截图级 UI 验证，但至少要运行目标 pytest 和 `git diff --check`。

## 验收标准

- Settings 页面能选择 `Auto`、`English`、`中文`。
- 选择结果能持久保存。
- `zh_CN` 启动时已迁移的界面文本显示中文。
- `en_US` 或翻译文件缺失时界面保持英文。
- 必要参数和数据语义仍保持英文。
- 中文译文读起来不像机翻，术语和标点保持一致。
