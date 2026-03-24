<!-- card-schema: {"card_name": "Card Group", "source_file": "src/NepTrainKit/ui/views/_card/card_group.py", "serialized_keys": ["card_list", "filter_card"]} -->

# 卡片组（Card Group）

`Group`: `Container`  
`Class`: `CardGroup`  
`Source`: `src/NepTrainKit/ui/views/_card/card_group.py`

## 功能说明
用于组织共享同一输入的多分支流程（card container）。组内卡片顺序执行但都读取同一输入数据，最终汇总分支输出；本身不直接做结构变换。

它最适合的场景是：把同一输入结构同时送入两条互不依赖的分支，再在组尾统一汇总输出。如果你更关心完整工作流而不是单个参数，请先看下面的“操作示例”。

:::{tip}
在软件中，直接将需要合并的卡片拖入卡片组中即可
:::

## 操作示例
### 场景：把同一输入结构同时送入两条互不依赖的分支，再在组尾统一汇总输出

**输入：** 一个已经准备好的母相结构，以及两条想并行执行的后续链路，例如“表面链”和“缺陷链”

**目标：** 避免重复导入同一输入，让多条分支共享同一批起始结构，同时在组内放一个过滤卡做局部筛选

**参数设置：**
- `card_list` 中依次放入两条或多条分支卡片
- `filter_card=true` 只在你明确要把该卡当作组内筛选器时开启

**输出：** 来自所有子卡片分支的汇总结果；每条分支共享同一输入，但各自独立生成结构

**怎么验证结果合理：**
- 确认组内每条分支都真的接收到了同一输入
- 检查过滤卡是否只在组内起作用，而不是被误当成新的生成卡
- 抽查不同分支输出标签，确保来源可区分

## 适用场景与不适用场景
- 数据症状 (Dataset symptom): 同一输入需要并行生成多类变体，顶层流程变得冗长且难维护。
- 目标任务 (Target objective): 把共享输入的多分支卡片收敛为一个容器并统一启停。
- 建议添加条件 (Add-it trigger): 多个分支需要共享同一输入并汇总输出。
- 不建议添加条件 (Avoid trigger): 需要前一卡输出驱动后一卡（严格串行依赖）时不应使用 Card Group。
> 物理提示 (Physics caution): 组内共享的是输入，不是每张卡的状态；调试时要按分支逐条验证，而不是只看汇总结果。

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
- 参数联动 / 生效条件: 组内每张子卡共享同一输入，但彼此不串联；顺序更多是组织和展示意义，而不是线性数据流。
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
- 参数联动 / 生效条件: 开启后当前卡会被当作组内筛选用途，而不是继续向下游充当普通生成卡。
- 怎么判断该开还是该关: 先用默认值跑小样本；只有当你能明确说明它会改变当前结果分布时，再主动偏离默认设置。
- 配置建议 (Practical note): 用于对组内汇总结果做可选筛选；当前不作为下游卡片输入源。若下游需要过滤结果，请在 Group 后串接独立过滤卡。

## 推荐预设（可直接复制 JSON）
### 基础容器（Baseline）
```json
{
  "class": "CardGroup",
  "check_state": true,
  "card_list": [],
  "filter_card": null
}
```


## 推荐组合
- Card Group(Atomic Perturb, Lattice Perturb, Shear Matrix Strain) -> 组外过滤链路: 组内先生成共享输入的多分支结果，再在组外统一清洗与采样。
- Card Group(Random Vacancy, Insert Defect) -> export: 适合从同一输入并行生成互补缺陷族；若需要严格依赖链，请拆回顶层顺序卡片。
- 容器卡适合组织并行分支，而不是替代真正的生成或过滤逻辑。

## 常见问题与排查
- 输出为空时，先检查组内子卡是否都可运行，以及是否有过滤卡把所有结果都筛掉。
- 若分支结果混淆，优先检查每条分支是否有清晰的 `Config_type` 或 group 标记。
- 容器卡不会改变子卡自身逻辑；它只组织输入共享和结果汇总。

## 输出标签 / 元数据变更
- 该卡片本身不新增专用 Config_type 标签。
- 在卡片 JSON 中保存嵌套 `card_list` 定义及可选 `filter_card`。

## 可复现性说明
- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。
- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。
