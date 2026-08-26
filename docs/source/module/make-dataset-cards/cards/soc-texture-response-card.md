<!-- card-schema: {"card_name": "SOC / Texture Response", "source_file": "src/NepTrainKit/ui/views/_card/soc_texture_response_card.py", "serialized_keys": ["params"]} -->

# SOC / 纹理响应（SOC / Texture Response）

**分类：** 磁性

## 功能说明

生成全局刚性自旋旋转或有限 q 纹理扫描。Bulk/Bloch、Interfacial/Cycloidal 和 General 是构型 preset，不是固定 Hamiltonian head；同组输出按有符号坐标排序并保留 manifest。

## 原理与公式

全局扫描对所有自旋施加同一个 $R(\theta)$，因此 $\mathbf S_i'\cdot\mathbf S_j'=\mathbf S_i\cdot\mathbf S_j$。螺旋相位使用 $\phi_i=\mathbf q\cdot\mathbf r_i+\phi_0$；正负 q 用于辨识奇偶响应，不做能量二阶中心差分。

## 操作示例

### 场景：模型无法区分 Bloch 与 cycloidal 手性

输入一个有明确晶胞和自旋幅值的超胞。界面选择与体系对称性匹配的 preset，给出可共格的 Cartesian q，并扫描 `-2,-1,0,1,2` 倍 q0。先核对 manifest 中的 q、plane normal、period 和 chirality，再对 DFT 能量或磁力响应做正负 q 奇偶分解。单条 qscan 最低点不能代替局域 pair response 验收。

## 参数说明

### Response Kind（response_kind）

`str`，默认 `Global anisotropy`。可选 `Global anisotropy`、`Bulk / Bloch`、`Interfacial / Cycloidal`、`General spiral`。

### Coordinate Scan（coordinate_scan）

`str`，默认 `-2,-1,0,1,2`。界面使用“最小值 / 最大值 / 步长”控件；全局扫描时明确标为度，q 扫描时明确标为基准 q 的有符号倍数。非等距旧扫描可使用自定义坐标列表。

### Rotation Axis（rotation_axis）

Cartesian 三维向量，默认 `[0,1,0]`。全局刚性旋转轴。

### Q Vector Cart（q_vector_cart）

Cartesian 三维向量，默认 `[0,0,0.1]`，单位 1/Å。界面将其拆为“传播方向”preset 和“基准 |q|”，避免用户手算三维分量；序列化和 manifest 仍保存完整 Cartesian q，并同时记录 reciprocal/fractional 表示。

### Plane Normal（plane_normal）

Cartesian 三维向量，默认 `[0,1,0]`。General spiral 的旋转平面法向。

### Surface Normal（surface_normal）

Cartesian 三维向量，默认 `[0,0,1]`。Cycloidal preset 与 q 一起定义旋转平面；不能与 q 平行。

### Cone Component（cone_component）

`float`，默认 `0`，范围 `[-1,1]`。沿 plane normal 的归一化锥分量。

### Phase Deg（phase_deg）

`float`，默认 `0` 度。纹理相位偏移；默认不做昂贵的多相位扫描。

### Include Time Reversal（include_time_reversal）

`bool`，默认关闭。为全局各向异性路径增加 $\mathbf S\to-\mathbf S$ 负对照组。

### Require Commensurate（require_commensurate）

`bool`，默认开启。q 在任一周期晶格矢量上不闭合时明确报错并提示扩大超胞，不生成有接缝的纹理。

### Max Outputs（max_outputs）

`int`，默认 `100`。只允许完整路径进入输出。

## 推荐预设

```json
{"class":"SOCTextureResponseCard","params":{"response_kind":"Global anisotropy","coordinate_scan":"-90,-45,0,45,90","rotation_axis":[0,1,0],"include_time_reversal":true,"max_outputs":20}}
```

## 推荐组合

- `Set Magnetic Moments → SOC / Texture Response`：建立 FM、AFM 或一般非共线参考纹理后整体旋转。
- `Super Cell → SOC / Texture Response`：先构造与 q 共格的超胞，再生成有限 q 数据。

## 常见问题

- q 不共格：改变 q 或沿报错的周期晶格方向扩胞；程序不会静默继续。
- cycloidal 报法向平行：surface normal 必须与 q 张成平面。
- 无 SOC 数据看不到各向异性：这是合理负对照，不应靠生成器制造能量差。

## 输出标签

`Config_type` 追加 `MagResponse(global_anisotropy|bulk_bloch|interfacial_cycloidal|general_spiral,branch)`。

## 可复现性

没有随机分支；q、phase、plane 和 supercell 信息完整进入 manifest。
