# 需求/脚本到卡片规格模板

按下面模板先产出规格，再写代码。

## 1. 输入信息

- 输入类型：需求描述 / 小脚本 / 现有卡片迁移
- 原始输入摘要：
- 目标数据处理动作：

## 2. 卡片基础定义

- 类名：
- `card_name`：
- 查找弹窗一句简述：
- `group`：
- `menu_icon`：
- `requires_input_dataset`：
- Operation 类型：`StructureOperation` / `DatasetOperation` / `GeneratorOperation`
- Core 模块：`lattice.py` / `structure.py` / `alloy.py` / `defect.py` / `magnetism.py` / `filter.py`

## 3. 卡片设计审查

- 用户任务（一句话说明输入 → 操作 → 输出）：
- 输入前提：
- 明确不负责的能力：
- 相邻/相似卡片：
- 不重复的理由：
- 是否混入了应拆分的生成、过滤或标签职责：
- 方向/晶面/层操作的参考系：
- 任意取向支持边界：
- 常用参数与高级参数如何分层：
- 哪些控件需要隐藏/禁用联动：
- 新增下拉项、分组、枚举和提示的翻译触点：

详细检查见 `card-design-review.md`。

## 4. 操作示例（先写这个）

在写代码前先把操作示例定下来。必须从训练集诊断出发：

- **模型故障现象：** 模型在什么任务上、什么指标变差了
- **根因诊断：** 训练集缺了什么数据导致这个问题
- **输入结构：** 具体是什么输入
- **预期改善：** 加入这批数据后，重训模型应该看到什么变化
- **参数设置：** 解决这个问题大概需要什么参数/量级
- **验证方法：** 怎么判断训练集质量确实改善了

## 5. 参数设计（Params + UI）

对每个参数填写：

- 参数名（代码 key）：
- Params 字段类型：
- UI 控件类型（SpinBoxUnitInputFrame / ComboBox / CheckBox / LineEdit）：
- 类型与范围：
- 默认值：
- 步长/小数位：
- 禁用语义（`0` / `None` / 空值 / 独立开关）：
- 单位与参考系：
- 使用建议（场景 + 量级，不要泛化模板句）：
- 参数联动 / 生效条件（仅在条件生效时写）：
- 该参数失效时控件隐藏还是禁用：
- 哪条测试证明它确实影响输出：

## 6. 业务逻辑设计

- Operation 方法：`run_structure(structure, params)` / `run_dataset(dataset, params)` / `generate(params)`
- 输入对象：
- 输出对象：`list[ase.Atoms]`
- 核心算法步骤（3-8 条）：
- 随机性与 seed 策略：
- 输出数量合同：严格数量 / 尽力生成 / 允许随机耗尽失败 / 过滤子集
- preview/summary：无 / 精确 / 估算；预计规模与线程策略：
- 异常处理策略（core 抛异常，UI 展示错误）：
- `Config_type` 标签策略：

## 7. 绑定与序列化

- 哪些 UI 字段参与运行时逻辑：
- `create_operation()` 返回什么：
- `get_params()` 如何从 UI 构造 Params：
- `set_params(params)` 如何恢复 UI：
- `to_dict` 字段清单：
- 是否写入 `"params"`：
- `from_dict` 默认值与兼容策略：
- 至少一个真实旧 JSON 的预期行为：
- 如果 `process_structure()` 存在，是否只做兼容委托：
  - Structure: `return self.create_operation().run_structure(structure, self.get_params())`
  - Dataset / Generator: 不新增伪 `process_structure()` 通路
- 是否覆盖 `run()`：必须为否

## 8. 测试风险矩阵

- 公开模式、关键分支和共享资源组合：
- 默认值、哨兵、非法值：
- 非正交 cell / 混合 PBC / 非默认取向（若适用）：
- 固定 seed / 多种子 / 最坏抽样（若适用）：
- 空输入、1 原子、NaN/Inf 等退化输入：
- 成功有输出 / 合法空输出 / 失败：
- preview 与 run 一致性或估算边界：
- 后台 preview 的防抖、最新结果和关闭生命周期：
- 注册、文档 URL、查找简述和翻译：
- 多出口产物一致性（若适用）：

详细检查见 `test-rigor-checklist.md`。

## 9. 接入与验证

- 注册改动：
- 查找弹窗与新增下拉框改动：
- 文档改动：
- 测试改动：
- operation 直接测试：
- UI 参数往返测试：
- 验证命令：

## 10. 评审检查点

- core operation 是否完全不依赖 UI？
- UI 是否只委托 operation，而不是实现算法？
- 是否存在“UI 有参数但逻辑没用”的死参数？
- 是否存在“逻辑有硬编码但 UI 没暴露”的隐含参数？
- 参数名/默认值在代码和文档是否一致？
- 老配置 JSON 能否安全加载？
- 是否有卡片数量、RNG 具体计数或 CI 绝对耗时等脆弱断言？
- 测试运行前后工作树是否保持同一状态？
