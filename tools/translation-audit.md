# 界面文案检查

在已安装项目测试依赖的环境中，从仓库根目录执行：

```sh
QT_QPA_PLATFORM=offscreen python tools/audit_translations.py --check --json /tmp/neptrainkit-text-audit.json
python -m pytest tests/test_translation_audit.py tests/test_i18n.py -q
```

Windows PowerShell 可先设置 `$env:QT_QPA_PLATFORM = "offscreen"`，再执行 Python 命令。

工具不修改源码或 TS 翻译；若生成的 QM 缺失或早于 TS，会先编译 QM。`--json` 可另存检查报告。报告包含文件、行号、原文、规则，以及适用时的模拟中文结果。默认扫描整个 `src` 下的 Python 文件，不把中英文文档、注释、测试数据、模型文件和第三方源码当作界面文案。

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

### QM 的生成与分发

Git 只跟踪 `.ts` 翻译源文件，`src/NepTrainKit/translations/*.qm` 是忽略的编译产物：

- 源码开发：加载中文界面时，如果 QM 缺失或 TS 更新，自动调用当前 Python 环境的 Qt 编译工具；未变化时不启动编译器。翻译审计也会执行同样的过期检查。
- wheel：setuptools 的 `build_py` 在构建目录中编译 QM 并装入 wheel。构建隔离环境通过 `PySide6-Essentials` 获得编译器，版本与应用使用的 PySide6 一致。
- sdist：包含 TS 和编译代码，排除 QM；从 sdist 构建 wheel 时重新生成。
- Windows 独立包：Nuitka 工作流在编译前生成 QM，再将翻译目录复制到分发目录。
- 安装后的 wheel 和独立包：直接加载随包的 QM，不在用户安装目录编译。

手动强制重编译用 `python tools/update_translations.py --no-lupdate`。编译失败会保留原 QM；构建和审计会失败并报告编译错误，源码 GUI 会记录警告。不要将 QM 再次加入 Git，也不要用“始终选择一侧”处理 TS 的真实翻译冲突。

## 已核实清单与回归门禁

`tools/translation_reviews/` 按模块保存候选的分类依据、整句模板及后续补修记录。源码记录采用对应模块路径，例如 `core/io/lammps_dump.json`；仅属于 TS 的记录放在 `catalog/<翻译上下文>.json`。检查工具递归读取各模块的 `retained_candidates`，保留经过复核的技术标识和已有兜底的异常。

拆分保留原始 454 条 `items`、125 条 `retained_candidates` 和 69 条 `followup_fixes`，包括原有 ID、理由和证据；这些数量只描述迁移时的记录，不限制后续新增。新增记录写到所属模块文件，不再追加到集中式 JSON。已有 ID 不重编号，新记录无需分配全局连续 ID；审计匹配不依赖 ID。审查记录是开发维护数据，不是运行时翻译，也不是可删除的临时报告。

```sh
python tools/audit_translations.py --check --require-reviewed --json /tmp/neptrainkit-text-audit.json
python -m pytest tests/test_runtime_message_review.py tests/test_translation_audit.py tests/test_i18n.py -q
```

CI 增加 `--require-reviewed`：只有路径、原文、上下文和中文模拟结果都与已核实记录一致，才计为已审查。新文案或中文结果变化会要求重新核实；不会因为文件曾进入清单而永久豁免。硬错误不能被审查记录豁免。

`ui/runtime_error_catalog.py` 是旧调用链的**完整消息模板**目录，处理异常已被转成字符串的情况；新卡片仍使用 `CardOperationError`。Qt 更新工具会保留该目录的翻译条目。界面在插入错误原因前调用 `translate_runtime_error`；未知诊断保留在日志中。原始异常类型、英文异常文本和参数值保持原有契约。

回归覆盖目录内的整句模板、诊断日志、中英文切换、含中文及英文关键字的路径，以及实际非法参数、审计原因和后台提示。静态扫描和这些测试不能替代所有窗口、所有第三方异常的人工验收。
