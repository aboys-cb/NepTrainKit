# 卡片文档编写规范

目标：文档可决策、可执行、可复现。

## 语言规范

- 正文中文
- 关键术语可保留英文（如 `seed`、`Config_type`、`sampling mode`）
- 字段名与 serialized key 保持英文

## 必备章节

- 功能说明
- 适用场景与不适用场景
- 输入前提
- 参数说明（完整）
- 推荐预设（可直接复制 JSON）
- 推荐组合
- 常见问题与排查
- 输出标签 / 元数据变更
- 可复现性说明

## 关键校验

- `Default` 必须与运行时代码默认值一致
- 空值必须写 `""` 或 `null`
- 禁止 `(empty)` 占位写法
- 字符串枚举与数字严格区分（如 `"111"` vs `111`）

## 提示块语法

对“强推荐流程”优先使用 MyST 提示块，而不是普通引用：
- 使用 `:::{tip} ... :::` 包裹“高通量示例”等关键提示，建议放在卡片页开头的“功能说明”中，便于先看到。

## 公式写法

- 仅在代码存在明确数学变换时添加 `### 关键公式 (Core equations)` 小节
- 公式使用 LaTeX 数学块（`$$ ... $$`）
- 公式符号要和代码参数保持可映射（如 `sxy/100`、`max_angle`、`min_distance_condition`）
- 没有稳定数学定义的卡片不强行补公式

## 维护命令

- `python tools/docs/audit_card_docs.py`
- `python -m sphinx -W -b html docs/source docs/build/html`
