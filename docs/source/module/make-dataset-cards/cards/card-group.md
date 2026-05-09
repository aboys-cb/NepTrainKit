<!-- card-schema: {"card_name": "Card Group", "source_file": "src/NepTrainKit/ui/views/_card/card_group.py", "serialized_keys": ["card_list", "filter_card"]} -->

# 卡片组（Card Group）

`Group`: `Container` | `Class`: `CardGroup`

## 功能说明

把共享同一输入的多张卡片装进一个容器里，并行执行后汇总输出。适合"同一个母相结构既要走表面链又要走缺陷链"这种分支场景。组内卡片互不依赖、各自独立读取组输入。

**不是做什么的：** 不是过滤器、不是生成卡、不支持组内卡片之间的串行依赖。串行依赖请放在组外顺序连接。

## 操作示例

### 场景：一份母相结构要同时生成表面 slab 和体相空位

你已经弛豫好一个 Ni 超胞，想同时做两件事：① 切不同取向的表面 slab；② 在体相内随机挖空位。两条路径互不依赖，但你不想分两次导数据。

**输入：** 一个弛豫好的 Ni 4×4×4 超胞

**目标：** 把 `Random Slab` 和 `Random Vacancy` 放入同一组，共享输入，并行生成两个分支，统一导出

**参数设置：**
- 将 `Random Slab` 卡片拖入组内，配置好 slab 参数
- 将 `Random Vacancy` 卡片拖入组内，配置好空位规则
- `card_list` 自动维护为 `[{RandomSlabCard...}, {RandomVacancyCard...}]`

**输出：** 两个分支的并集——一批不同取向表面 slab + 一批不同空位结构，各自带独立标签

**怎么验证：** 抽查输出，确认有明确不同标签来源（`Slab(...)` vs `Vac(...)`）。如果只有一种标签，检查组内卡片是否都勾选了 `check_state`。

### 什么时候用、什么时候不用

**用：**
- 同一输入需要同时喂给多张彼此无依赖的卡片
- 多条分支需要统一启停和导出

**不用：**
- 卡片之间有前后依赖（A 的输出是 B 的输入）→ 放在组外串行连接
- 只需要单张卡 → 不需要组

## 参数说明

### `card_list`

组内的子卡片列表，由界面拖入操作自动维护，不需要你手动编辑。每项是一个完整的卡片配置 dict（含 class、check_state、params 等字段），子卡片共享组输入、各自独立运行。

### `filter_card`

dict 或 null，默认 null。选填：在组内末尾挂一张过滤卡（如 `FPS Filter`），对并行生成的汇总结果做统一筛选。留 null 表示不启用组内过滤。

> 组内过滤只作用于组输出，不进入下游卡片的输入流。如果你需要在组外继续处理，把过滤卡串在组后。

## 推荐预设

### 空容器（只需要拖入卡片）
```json
{
  "class": "CardGroup",
  "check_state": true,
  "card_list": [],
  "filter_card": null
}
```

### 带组内过滤（先并行生成，再 FPS 筛）
```json
{
  "class": "CardGroup",
  "check_state": true,
  "card_list": [
    {"class": "RandomSlabCard", "check_state": true, "...": "..."},
    {"class": "RandomVacancyCard", "check_state": true, "...": "..."}
  ],
  "filter_card": {"class": "FPSFilterDataCard", "check_state": true, "nep_path": "...", "n_samples": 50}
}
```

## 推荐组合

- `Card Group(Random Slab, Insert Defect)` → `FPS Filter`：并行生成表面+缺陷 → 组外统一筛选
- `Card Group(Atomic Perturb, Lattice Perturb)` → export：同一输入同时补充坐标噪声+晶格噪声
- 如果两条分支有严格依赖，拆回顶层串行：`Super Cell → Random Slab → Insert Defect`

## 常见问题

**组输出为空。** 检查组内每张子卡片是否勾选了 `check_state`、各自配置是否合法。

**分支结果混在一起分不清。** 检查每张子卡片是否产出了不同的 `Config_type` 标签。组本身不区分来源——区分靠子卡片的标签。

**组内过滤卡把结果全筛掉了。** `filter_card` 的过滤参数可能过严（如 `n_samples` 设太大但输入很少）。先调松过滤参数，或去掉组内过滤改为组外过滤。

**想在组内串行。** 不支持。组内所有卡片共享同一输入、并行执行。串行依赖请拆到组外。

## 输出标签

该卡片本身不写入 `Config_type` 标签。标签由组内子卡片各自产生。

## 可复现性

无自身随机性。可复现性取决于组内子卡片的 seed 配置。
