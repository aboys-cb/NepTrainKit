# 界面文案检查

在已安装项目测试依赖的环境中，从仓库根目录执行：

```sh
QT_QPA_PLATFORM=offscreen python tools/audit_translations.py --check --json /tmp/neptrainkit-text-audit.json
python -m pytest tests/test_translation_audit.py tests/test_i18n.py -q
```

Windows PowerShell 可先设置 `$env:QT_QPA_PLATFORM = "offscreen"`，再执行 Python 命令。

工具只读取源码及翻译目录；只有 `--json` 指定的报告会被写入。报告包含文件、行号、原文、规则，以及适用时的模拟中文结果。默认扫描整个 `src` 下的 Python 文件，不把中英文文档、注释、测试数据、模型文件和第三方源码当作界面文案。

## 错误与候选

`--check` 在确认存在以下错误时返回非零退出码：

- 活跃 TS 条目为空或未完成，Python／Qt 占位符或格式规格不一致。
- 源码中的字面量翻译模板缺失，包括 `CardOperationError` 的模板。
- 先格式化再翻译，例如 `self.tr(f"Invalid {name}")`。应先翻译模板再 `.format()`。
- 已编译 QM 的实际结果与 TS 不一致。仅有 TS 文件不代表运行时已更新。

以下结果是人工审查候选，不会令检查失败：

- 直接传入常见控件 setter 的文字；少量已确认的产品名、元素符号和表达式语法有精确豁免。
- 翻译目录、原始消息或异常经现有中文翻译后，仍包含英文句式或英文残句的结果。

动态表达式用 `⟦value⟧` 代替，扫描不执行表达式。内部异常未必会显示到界面，因此候选数量不是界面缺陷数量；不应通过批量替换单词来消除候选。

## 覆盖边界

静态扫描不能证明全仓没有混排。它不执行任意控制流，也不会追踪变量到最终控件、自动展开所有控件构造参数、生成真实后端异常或验证截图。隐式 `self.tr` 的源码检查允许继承上下文中的同名模板；显式 `QCoreApplication.translate` 则按指定上下文检查。复数条目检查各翻译形式的占位符，QM 比对暂跳过复数。

确认的界面缺陷应补真实场景的中英文测试。文件名、模型名、用户输入和错误码应原样保留。卡片错误使用 `CardOperationError(code, template, **values)`；技术参数作为值插入，不扩充英文单词替换表。

## 更新翻译

```sh
python tools/update_translations.py --no-lrelease
# 在 TS 中填写新增中文翻译
python tools/update_translations.py --no-lupdate
python tools/audit_translations.py --check --json /tmp/neptrainkit-text-audit.json
```

更新工具还会通过 AST 提取结构化卡片异常及 `self.tr`、`_tr`、显式上下文翻译调用，补充 Qt lupdate 漏掉的 f-string 内部调用。CI 会运行检查并保存完整 JSON 报告；未解决候选仍保留在报告中。

## 已核实清单与回归门禁

`tools/translation_review.json` 保存原始 454 条候选的分类依据、整句模板及后续补修记录。`retained_candidates` 保留经过复核的技术标识和已有兜底的异常，不删除原始扫描结果。

```sh
python tools/audit_translations.py --check --require-reviewed --json /tmp/neptrainkit-text-audit.json
python -m pytest tests/test_runtime_message_review.py tests/test_translation_audit.py tests/test_i18n.py -q
```

CI 增加 `--require-reviewed`：只有路径、原文、上下文和中文模拟结果都与已核实记录一致，才计为已审查。新文案或中文结果变化会要求重新核实；不会因为文件曾进入清单而永久豁免。硬错误不能被审查记录豁免。

`ui/runtime_error_catalog.py` 是旧调用链的**完整消息模板**目录，处理异常已被转成字符串的情况；新卡片仍使用 `CardOperationError`。Qt 更新工具会保留该目录的翻译条目。界面在插入错误原因前调用 `translate_runtime_error`；未知诊断保留在日志中。原始异常类型、英文异常文本和参数值保持原有契约。

回归覆盖目录内的整句模板、诊断日志、中英文切换、含中文及英文关键字的路径，以及实际非法参数、审计原因和后台提示。静态扫描和这些测试不能替代所有窗口、所有第三方异常的人工验收。
