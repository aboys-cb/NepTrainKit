<!-- card-schema: {"card_name": "Card Group", "source_file": "src/NepTrainKit/ui/views/_card/card_group.py", "serialized_keys": ["card_list", "filter_card"]} -->

# 卡片组（Card Group）

`Group`: `Container`  
`Class`: `CardGroup`  
`Source`: `src/NepTrainKit/ui/views/_card/card_group.py`

## 功能说明
用于组织共享同一输入的多分支流程（card container）。组内卡片顺序执行但都读取同一输入数据，最终汇总分支输出；本身不直接做结构变换。

### 快速上手
最小可运行示例：在 Card Group 内放入两张互不依赖的分支卡片，共享同一输入运行后核对汇总输出条数。

:::{tip}
高通量示例：先在组内组织共享输入的多分支生成，再在组外串接清洗/采样链路（例如 NEP89 清洗与 FPS），避免把依赖链写进组内。
:::


## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 同一输入需要并行生成多类变体，顶层流程变得冗长且难维护。
- 目标任务 (Target objective): 把共享输入的多分支卡片收敛为一个容器并统一启停。
- 建议添加条件 (Add-it trigger): 多个分支需要共享同一输入并汇总输出。
- 不建议添加条件 (Avoid trigger): 需要前一卡输出驱动后一卡（严格串行依赖）时不应使用 Card Group。


## 输入前提
- 组内卡片应彼此独立且共享同一输入；若有依赖链，请放在组外顺序执行。
- 先在小规模数据上验证每个分支输出，再合并进同一个 Card Group。


## 参数说明（完整）
### `card_list` (Card List)
- UI Label: `Card List`
- 字段映射 (Field mapping): 序列化键 `card_list` <-> 界面标签 `Card List`。
- 控件标签 (Caption): `Card List`。
- 控件解释 (Widget): 按字段类型解析。
- 类型/范围 (Type/Range): list
- 默认值 (Default): `[]`
- 含义 (Meaning): 组内卡片列表 (card list)。
- 对输出规模/物理性的影响: 同一输入的分支集合，按布局顺序依次运行并汇总输出。
- 配置建议 (Practical note): 建议只放共享同一输入且互不依赖的分支卡片；若存在严格前后依赖，请移到组外顺序执行。

### `filter_card` (Filter Card)
- UI Label: `Filter Card`
- 字段映射 (Field mapping): 序列化键 `filter_card` <-> 界面标签 `Filter Card`。
- 控件标签 (Caption): `Filter Card`。
- 控件解释 (Widget): 按字段类型解析。
- 类型/范围 (Type/Range): dict
- 默认值 (Default): `null`
- 含义 (Meaning): 组内过滤节点 (filter card)。
- 对输出规模/物理性的影响: 对组内汇总结果执行可选筛选；当前不作为下游卡片输入源。
- 配置建议 (Practical note): 用于对组内汇总结果做可选筛选；当前不作为下游卡片输入源。若下游需要过滤结果，请在 Group 后串接独立过滤卡。


## 推荐预设（可直接复制 JSON）
### 保守（Safe）
```json
{
  "class": "CardGroup",
  "check_state": true,
  "card_list": [],
  "filter_card": {}
}
```

### 平衡（Balanced）
```json
{
  "class": "CardGroup",
  "check_state": true,
  "card_list": [],
  "filter_card": {}
}
```

### 激进/探索（Aggressive/Exploration）
```json
{
  "class": "CardGroup",
  "check_state": true,
  "card_list": [],
  "filter_card": {}
}
```


## 推荐组合
- Card Group(Atomic Perturb, Lattice Perturb, Shear Matrix Strain) -> 组外过滤链路: 组内先生成共享输入的多分支结果，再在组外统一清洗与采样。
- Card Group(Random Vacancy, Insert Defect) -> export: 适合从同一输入并行生成互补缺陷族；若需要严格依赖链，请拆回顶层顺序卡片。


## 常见问题与排查
- 误把组内当串行流水线：确认组内每张卡读入的都是同一 `dataset`。
- 汇总结果不符合预期：核对 `card_list` 中启用状态和分支顺序。
- 过滤后结果为空：核对 `filter_card` 条件是否过严，并确认它仅作用于组内导出链路。


## 输出标签 / 元数据变更
- 该卡片本身不新增专用 Config_type 标签。
- 在卡片 JSON 中保存嵌套 `card_list` 定义及可选 `filter_card`。


## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
