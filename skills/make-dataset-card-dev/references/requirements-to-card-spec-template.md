# 需求/脚本到卡片规格模板

按下面模板先产出规格，再写代码。

## 1. 输入信息

- 输入类型：需求描述 / 小脚本 / 现有卡片迁移
- 原始输入摘要：
- 目标数据处理动作：

## 2. 卡片基础定义

- 类名：
- `card_name`：
- `group`：
- `menu_icon`：
- `requires_input_dataset`：
- Operation 类型：`StructureOperation` / `DatasetOperation` / `GeneratorOperation`
- Core 模块：`lattice.py` / `structure.py` / `alloy.py` / `defect.py` / `magnetism.py` / `filter.py`

## 3. 参数设计（Params + UI）

对每个参数填写：

- 参数名（代码 key）：
- Params 字段类型：
- UI 控件类型（SpinBoxUnitInputFrame / ComboBox / CheckBox / LineEdit）：
- 类型与范围：
- 默认值：
- tooltip：
- 在 `params` 中的 key：
- 是否保留旧 `to_dict` key：

## 4. 业务逻辑设计

- Operation 方法：`run_structure(structure, params)` / `run_dataset(dataset, params)` / `generate(params)`
- 输入对象：
- 输出对象：`list[ase.Atoms]`
- 核心算法步骤（3-8 条）：
- 随机性与 seed 策略：
- 异常处理策略（core 抛异常，UI 展示错误）：
- `Config_type` 标签策略：

## 5. 绑定与序列化

- 哪些 UI 字段参与运行时逻辑：
- `create_operation()` 返回什么：
- `get_params()` 如何从 UI 构造 Params：
- `set_params(params)` 如何恢复 UI：
- `to_dict` 字段清单：
- 是否写入 `"params"`：
- `from_dict` 默认值与兼容策略：
- 如果 `process_structure()` 存在，是否只做兼容委托：
  - Structure: `return self.create_operation().run_structure(structure, self.get_params())`
  - Dataset / Generator: 不新增伪 `process_structure()` 通路
- 是否覆盖 `run()`：必须为否

## 6. 接入与验证

- 注册改动：
- 文档改动：
- 测试改动：
- operation 直接测试：
- UI 参数往返测试：
- 验证命令：

## 7. 评审检查点

- core operation 是否完全不依赖 UI？
- UI 是否只委托 operation，而不是实现算法？
- 是否存在“UI 有参数但逻辑没用”的死参数？
- 是否存在“逻辑有硬编码但 UI 没暴露”的隐含参数？
- 参数名/默认值在代码和文档是否一致？
- 老配置 JSON 能否安全加载？
