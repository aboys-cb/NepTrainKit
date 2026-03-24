# 需求/脚本到卡片规格模板

按下面模板先产出规格，再写代码。

## 1. 输入信息

- 输入类型：需求描述 / 小脚本
- 原始输入摘要：
- 目标数据处理动作：

## 2. 卡片基础定义

- 类名：
- `card_name`：
- `group`：
- `menu_icon`：
- `requires_input_dataset`：

## 3. 参数设计（UI）

对每个参数填写：

- 参数名（代码 key）：
- UI 控件类型（SpinBoxUnitInputFrame / ComboBox / CheckBox / LineEdit）：
- 类型与范围：
- 默认值：
- tooltip：
- 在 `to_dict` 中的 key：

## 4. 业务逻辑设计

- `process_structure` 输入：
- `process_structure` 输出：
- 核心算法步骤（3-8 条）：
- 随机性与 seed 策略：
- 异常处理策略：
- `Config_type` 标签策略：

## 5. 绑定与序列化

- 哪些 UI 字段参与运行时逻辑：
- `to_dict` 字段清单：
- `from_dict` 默认值与兼容策略：

## 6. 接入与验证

- 注册改动：
- 文档改动：
- 测试改动：
- 验证命令：

## 7. 评审检查点

- 是否存在“UI 有参数但逻辑没用”的死参数？
- 是否存在“逻辑有硬编码但 UI 没暴露”的隐含参数？
- 参数名/默认值在代码和文档是否一致？
- 老配置 JSON 能否安全加载？
