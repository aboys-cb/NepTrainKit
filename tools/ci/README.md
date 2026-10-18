# Windows Nuitka 构建与依赖审计

## 复用编译缓存

发布 release（`release.published`）会自动在默认分支 `master` 派发一次
Windows 构建，源码使用该 release tag；验证通过后上传 ZIP 到对应 release。
release 入口只负责派发，构建结果应查看 Actions 中随后出现的
`workflow_dispatch` run。这样缓存统一读写默认分支，避免不同 release tag
之间无法恢复缓存，也避免只在 tag 下保存、不能供下一个版本复用的问题。

手动修复已有 release 时，在 **Run workflow** 中选择默认分支 `master`，
`ref_name` 填修复后的分支或 commit，`publish_release` 选 true，
`release_tag` 填已有版本，例如 `v3.1.5`。源码必须是原 tag 的后代；
工作流沿用 release 版本号并覆盖对应 ZIP，不移动 tag。只验证打包时保留
`publish_release=false`，ZIP 仍作为 Actions artifact 上传。

工作流固定 Nuitka 和 Nuitka-Action 到已成功构建的 revision，通过
`NUITKA_CACHE_DIR` 保存编译缓存。缓存前缀包含架构、Python 完整版本和
Nuitka revision；每个 run/attempt 保存一个新快照，后续从相同前缀的最新
快照恢复。源码和依赖变化交给 clcache 按编译输入判断是否可复用。
缓存使用独立 restore/save 步骤，Nuitka 成功后立即保存；后续资源检查、
GUI 启动或 release 上传失败不再丢失已完成的编译缓存。
`disable-cache: true` 只关闭 Nuitka-Action 自带的缓存包装步骤，实际编译
缓存仍由上述目录启用。

新缓存命名空间的首次运行是冷构建。Python/Nuitka 升级会建立新命名空间，
Windows runner 的 MSVC 更新也可能降低对象命中率；固定 Nuitka 不等于固定
整个 runner 镜像。GitHub 的缓存清理、容量限制同样可能导致冷构建。

验证时比较相同源码和依赖的冷、热两次构建：

1. 查看缓存恢复步骤是否找到归档。
2. 查看 `Nuitka-Scons` 日志中的 clcache hits/misses；归档恢复不等于对象命中。
3. 比较 C 编译/链接时间及整个 job 时间，并确认原生模块、GUI 启动和运行时检查通过。

每次构建上传 `Nuitka-diagnostics-<run_id>-<attempt>`，包含源码 commit、
Python 版本、已解析的依赖版本和 Nuitka compilation report。报告用于检查
实际包含的模块；不要用 pip 安装列表推断最终打包内容。

数据集 catalog JSON 按包内路径包含，并在编译后校验内容与源码一致。
软件 OpenGL DLL 取自同一套 PySide6 安装，GUI smoke 使用软件 OpenGL；
编译前检查源资源，失败时上传启动日志，保留原有原生模块和运行时检查。
Windows wheel 的 DLL 位于 PySide6 包目录，定位时优先检查该目录，再检查
Qt 报告的 bin 目录；预检查加载成功后记录确切路径，打包复制复用该路径。

参考：[GitHub 缓存作用域](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache)、
[GITHUB_TOKEN 派发 workflow](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)、
[Nuitka 缓存说明](https://nuitka.net/user-documentation/tips.html)。

## SciPy / Matplotlib 使用清单

以下为当前源码审计，路径相对 `src/NepTrainKit/`。未改变算法或移除依赖。

| 位置 | API | 用途 |
| --- | --- | --- |
| `core/alloy.py` | `stats.qmc.Sobol`, `random_base2` | 合金成分单纯形采样 |
| `core/cards/lattice.py` | `Sobol`, `random`, `MAXDIM` | 晶胞缩放和原子扰动采样 |
| `core/cards/defect.py` | `Sobol`, `random`, `MAXDIM` | 空位配置采样 |
| `core/audit/neighbor_scan.py` | `spatial.cKDTree`, `query_pairs` | 原子接触/距离检查，包括周期边界 |
| `core/audit/phase_sketch.py` | `cKDTree.query`, `special.eval_legendre` | 近邻与键取向序参量 |
| `core/structure.py` | `sparse.csgraph.connected_components` | 连通原子簇识别 |
| `core/io/sampling_states.py` | `cKDTree.query`, `query_ball_point`; `coo_matrix`, `csr_matrix`, `hstack`; `connected_components`, `minimum_spanning_tree`; `optimize.milp`, `Bounds`, `LinearConstraint` | 采样状态识别、稀疏邻域图和覆盖选择优化 |
| `core/io/distribution.py` | `stats.gaussian_kde`, `evaluate` | 分布图核密度估计；普通正态曲线已有内部 NumPy 实现 |
| `core/magnetic_response.py::write_response_audit` | Matplotlib `subplots`, `plot`, `set`, `legend`, `grid`, `tight_layout`, `savefig`, `close` | 导出能量/磁响应双面板 PNG，另存 JSON/CSV |

SciPy 已承担多个核心算法。只替换 Legendre 多项式或连通分量无法移除整体
依赖；用简单随机数替代 Sobol、用贪心替代 MILP 还会改变采样语义。
因此本轮保留 SciPy，不自行重写这些算法。

Matplotlib 的项目内直接使用只有上述 PNG 导出；当前检索到的项目内调用方
是对应测试，主要 GUI 图形使用 VisPy/pyqtgraph。复用已有 Qt 绘图设施替代
该导出是可行候选，但应保留无显示环境导出、NaN 曲线处理、坐标轴和图例。
仅删掉直接 import 不足以证明它能从 Windows 包中移除。

本地 ASE 3.28.0 元数据同时声明 SciPy 和 Matplotlib 依赖，ASE neighborlist
实际使用 SciPy。ASE 图片导出、VisPy/pyqtgraph 的部分色图接口也引用
Matplotlib；这些包含可选或延迟导入。安装依赖、静态引用和运行时必要依赖
需分别判断。后续若裁剪 Matplotlib，先用 compilation report 确认包含原因，
再验证实际使用的 ASE I/O、色图和 PNG 导出路径，最后比较包体和冷构建耗时。
