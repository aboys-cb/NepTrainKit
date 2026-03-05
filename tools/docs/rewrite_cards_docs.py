"""重写 Make Dataset 卡片文档为中文主叙事风格。

该脚本更新 docs/source/module/make-dataset-cards/cards 下所有卡片页，
保留 metadata 与章节顺序，但统一为中文章节标题，并重写关键段落：
- 功能说明
- 适用场景与不适用场景
- 输入前提
- 参数说明（完整）
- 常见问题与排查
- 可复现性说明
"""

from __future__ import annotations

import json
import importlib.util
import inspect
import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from NepTrainKit.core import CardManager
from NepTrainKit.ui.views import _card as _registered_cards  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
CARD_DOC_DIR = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "cards"

SECTION_ORDER = [
    "功能说明",
    "适用场景与不适用场景",
    "输入前提",
    "参数说明（完整）",
    "推荐预设（可直接复制 JSON）",
    "推荐组合",
    "常见问题与排查",
    "输出标签 / 元数据变更",
    "可复现性说明",
]

SECTION_ALIASES: dict[str, list[str]] = {
    "功能说明": ["功能说明", "What this card does"],
    "适用场景与不适用场景": ["适用场景与不适用场景", "When to use / when not to use"],
    "输入前提": ["输入前提", "Input prerequisites"],
    "参数说明（完整）": ["参数说明（完整）", "Control reference (complete)"],
    "推荐预设（可直接复制 JSON）": [
        "推荐预设（可直接复制 JSON）",
        "Recommended presets (copy-paste JSON)",
    ],
    "推荐组合": ["推荐组合", "Recommended combinations"],
    "常见问题与排查": ["常见问题与排查", "Failure modes & troubleshooting"],
    "输出标签 / 元数据变更": ["输出标签 / 元数据变更", "Output tags / metadata changes"],
    "可复现性说明": ["可复现性说明", "Reproducibility notes"],
}

CARD_TITLE_CN: dict[str, str] = {
    "card-group.md": "卡片组（Card Group）",
    "cell-scaling-card.md": "晶格扰动（Lattice Perturb）",
    "cell-strain-card.md": "晶格应变（Lattice Strain）",
    "composition-sweep-card.md": "成分扫描（Composition Sweep）",
    "conditional-replace-card.md": "条件替换（Conditional Replace）",
    "crystal-prototype-builder-card.md": "晶体原型构建（Crystal Prototype Builder）",
    "fps-filter-card.md": "FPS 过滤（FPS Filter）",
    "group-label-card.md": "分组标记（Group Label）",
    "interstitial-adsorbate-card.md": "插隙/吸附缺陷（Insert Defect）",
    "layer-copy-card.md": "层复制（Layer Copy）",
    "magmom-rotation-card.md": "磁矩旋转（Magmom Rotation）",
    "magnetic-order-card.md": "磁有序初始化（Magnetic Order）",
    "organic-mol-config-pbc-card.md": "有机构象采样（Organic Mol Config）",
    "perturb-card.md": "原子扰动（Atomic Perturb）",
    "random-doping-card.md": "随机掺杂（Random Doping）",
    "random-occupancy-card.md": "随机占位（Random Occupancy）",
    "random-slab-card.md": "随机表面切片（Random Slab）",
    "random-vacancy-card.md": "随机空位（Random Vacancy）",
    "shear-angle-card.md": "剪切角应变（Shear Angle Strain）",
    "shear-matrix-card.md": "剪切矩阵应变（Shear Matrix Strain）",
    "stacking-fault-card.md": "层错构型（Stacking Fault）",
    "super-cell-card.md": "超胞生成（Super Cell）",
    "vacancy-defect-card.md": "空位缺陷生成（Vacancy Defect Generation）",
    "vibration-perturb-card.md": "振动模态扰动（Vib Mode Perturb）",
}


CARD_PROFILE: dict[str, dict[str, list[str] | str]] = {
    "card-group.md": {
        "does": "用于组织共享同一输入的多分支流程（card container）。组内卡片顺序执行但都读取同一输入数据，最终汇总分支输出；本身不直接做结构变换。",
        "when": [
            "数据症状 (Dataset symptom): 同一输入需要并行生成多类变体，顶层流程变得冗长且难维护。",
            "目标任务 (Target objective): 把共享输入的多分支卡片收敛为一个容器并统一启停。",
            "建议添加条件 (Add-it trigger): 多个分支需要共享同一输入并汇总输出。",
            "不建议添加条件 (Avoid trigger): 需要前一卡输出驱动后一卡（严格串行依赖）时不应使用 Card Group。",
        ],
        "prereq": [
            "组内卡片应彼此独立且共享同一输入；若有依赖链，请放在组外顺序执行。",
            "先在小规模数据上验证每个分支输出，再合并进同一个 Card Group。",
        ],
        "fail": [
            "误把组内当串行流水线：确认组内每张卡读入的都是同一 `dataset`。",
            "汇总结果不符合预期：核对 `card_list` 中启用状态和分支顺序。",
            "过滤后结果为空：核对 `filter_card` 条件是否过严，并确认它仅作用于组内导出链路。",
        ],
    },
    "cell-scaling-card.md": {
        "does": "对晶胞尺度与坐标做轻量随机扰动（lattice perturbation），用于补充近平衡态的几何变化样本。",
        "when": [
            "数据症状 (Dataset symptom): 模型对小体积变化或轻微几何噪声敏感。",
            "目标任务 (Target objective): 扩展近似热涨落区域的结构覆盖。",
            "建议添加条件 (Add-it trigger): 需要比静态结构更密集的微扰样本。",
            "不建议添加条件 (Avoid trigger): 结构已经明显不稳定或接近相变边界。",
        ],
        "prereq": [
            "输入结构应先完成基本几何清洗（无明显重叠）。",
            "若输入含有机分子，必须开启 `organic` 以启用团簇识别和刚性移动。",
        ],
        "fail": [
            "出现短键/重叠：先降低 `scaling_condition` 幅度。",
            "样本膨胀过快：保持幅度，优先减少 `num_condition` 或下游加 FPS。",
        ],
    },
    "cell-strain-card.md": {
        "does": "按轴向组合扫描应变（uniaxial/biaxial/triaxial/isotropic），系统构建应力-应变覆盖数据。",
        "when": [
            "数据症状 (Dataset symptom): 弹性相关预测不稳，轴向响应泛化差。",
            "目标任务 (Target objective): 构建可解释的应变网格数据。",
            "建议添加条件 (Add-it trigger): 需要比较不同应变路径的模型表现。",
            "不建议添加条件 (Avoid trigger): 仅需要局部坐标噪声而非系统应变。",
        ],
        "prereq": [
            "先确定 `engine_type` 与研究问题匹配。",
            "控制步长与范围，防止组合数失控。",
        ],
        "fail": [
            "输出数量过大：减少轴组合或增大步长。",
            "高应变导致异常结构：收窄 `x/y/z_range` 极值。",
        ],
    },
    "composition-sweep-card.md": {
        "does": "在元素池和元数约束下生成成分设计空间（composition design space），用于合金候选前置展开。",
        "when": [
            "数据症状 (Dataset symptom): 元素组合覆盖单一，跨成分迁移误差高。",
            "目标任务 (Target objective): 先覆盖成分空间，再下游做占位/掺杂。",
            "建议添加条件 (Add-it trigger): 需要二元到多元合金系统化采样。",
            "不建议添加条件 (Avoid trigger): 任务只关注单一固定化学计量。",
        ],
        "prereq": [
            "明确元素池、元数上限和预算上限。",
            "先小规模验证 `method` 与 `budget_mode` 的分布行为。",
        ],
        "fail": [
            "组合数远超预算：收窄 `order` 并降低 `max_outputs`。",
            "成分分布偏置明显：调整 `method` 与 `budget_mode`。",
        ],
    },
    "conditional-replace-card.md": {
        "does": "按空间表达式对目标元素执行条件替换（conditional replacement），构建区域选择性化学改性样本。",
        "when": [
            "数据症状 (Dataset symptom): 需要只在表层/局域区域替换元素。",
            "目标任务 (Target objective): 增强局域化学环境变化覆盖。",
            "建议添加条件 (Add-it trigger): 可以用 `x/y/z` 明确写出作用区域。",
            "不建议添加条件 (Avoid trigger): 仅需全局替换，Random Doping 更直接。",
        ],
        "prereq": [
            "先验证 `replacements` 语法正确。",
            "先用 `condition=all` 验证路径，再收紧条件。",
        ],
        "fail": [
            "替换数为 0：通常是 `condition` 未命中目标元素。",
            "替换比例偏差大：检查 `mode`（Random vs Exact ratio）。",
        ],
    },
    "crystal-prototype-builder-card.md": {
        "does": "按晶型原型和晶格常数范围生成标准晶体起始结构，快速搭建可控的基础结构库。",
        "when": [
            "数据症状 (Dataset symptom): 缺少标准原型结构，训练集拓扑单一。",
            "目标任务 (Target objective): 构建 clean prototype baseline。",
            "建议添加条件 (Add-it trigger): 需要系统对比 fcc/bcc/hcp 等晶型。",
            "不建议添加条件 (Avoid trigger): 已有充分真实结构且无需原型补充。",
        ],
        "prereq": [
            "确认 `lattice` 与元素组合物理可行。",
            "设好 `max_outputs` 避免网格过密。",
        ],
        "fail": [
            "生成失败：检查 `a_range`、`rep` 是否产生非法晶胞。",
            "规模过大：降低 `max_atoms` 或启用自动扩胞策略。",
        ],
    },
    "fps-filter-card.md": {
        "does": "基于特征距离执行最远点采样（FPS），用于在完成物理清洗后压缩冗余并保留多样性。",
        "when": [
            "数据症状 (Dataset symptom): 数据量大但冗余高，训练收益下降。",
            "目标任务 (Target objective): 在删除非物理结构后保留代表性结构分布。",
            "建议添加条件 (Add-it trigger): 已完成 `nep89` 预测筛查并剔除不合理结构。",
            "不建议添加条件 (Avoid trigger): 仍处于样本生成早期或尚未完成物理清洗。",
        ],
        "prereq": [
            "先导出 xyz 并在第一个模块用 `nep89` 预测，删除不合理结构后再执行 FPS。",
            "确认描述符模型路径 `nep_path` 有效。",
            "先在小集试 `min_distance_condition` 对保留率影响。",
        ],
        "fail": [
            "先选中非物理结构：检查是否跳过了 `nep89` 清洗步骤。",
            "样本保留过少：降低最小距离阈值。",
            "去重不明显：适度提高阈值并确认特征有效。",
        ],
    },
    "group-label-card.md": {
        "does": "为结构生成 `group` 标签数组，支撑后续 AFM 分组、局域替换和规则型筛选。",
        "when": [
            "数据症状 (Dataset symptom): 下游操作需要分组，但当前无 group 标签。",
            "目标任务 (Target objective): 建立可复用的子晶格/区域标签语义。",
            "建议添加条件 (Add-it trigger): 下游存在 Magnetic Order 或 rules+group 操作。",
            "不建议添加条件 (Avoid trigger): 全流程不依赖 group 过滤。",
        ],
        "prereq": [
            "统一 `group_a/group_b` 命名规范。",
            "确认是否允许覆盖已有 group（`overwrite`）。",
        ],
        "fail": [
            "分组不符合预期：检查 `kvec` 与晶向定义。",
            "下游读不到分组：核对是否被覆盖或未写入数组。",
        ],
    },
    "interstitial-adsorbate-card.md": {
        "does": "在体相或表面插入额外原子/片段（interstitial or adsorbate insertion），采样缺陷与吸附构型。",
        "when": [
            "数据症状 (Dataset symptom): 模型对插层/吸附位点预测误差高。",
            "目标任务 (Target objective): 补充间隙位和吸附态样本。",
            "建议添加条件 (Add-it trigger): 研究扩散、吸附、非本征缺陷。",
            "不建议添加条件 (Avoid trigger): 仅关注完美晶体基态。",
        ],
        "prereq": [
            "`species` 必须是合法输入并与体系匹配。",
            "根据密度先设保守 `min_distance`。",
        ],
        "fail": [
            "插入成功率低：减小 `min_distance` 或增加 `max_attempts`。",
            "速度过慢：先降 `structure_count` 再逐步放宽。",
        ],
    },
    "layer-copy-card.md": {
        "does": "复制层并按 `dz_expr` 施加位移调制，生成层间错位、起伏和堆叠变化数据。",
        "when": [
            "数据症状 (Dataset symptom): 层状体系样本单一，层间自由度覆盖不足。",
            "目标任务 (Target objective): 增强层间几何变化与形貌多样性。",
            "建议添加条件 (Add-it trigger): 研究二维材料、多层异质结、层间耦合。",
            "不建议添加条件 (Avoid trigger): 非层状体相体系。",
        ],
        "prereq": [
            "先在单帧验证 `dz_expr` 与 `params`。",
            "按边界条件选择 `wrap/extend_cell_z/extra_vacuum`。",
        ],
        "fail": [
            "层间冲突：收窄 `z_range` 并增大 `distance`。",
            "边界异常：核对 `wrap` 与 `extend_cell_z` 配置。",
        ],
    },
    "magmom-rotation-card.md": {
        "does": "旋转指定元素的磁矩方向并可扰动模长，构建连续磁构型邻域数据。",
        "when": [
            "数据症状 (Dataset symptom): 非共线磁方向相关误差高。",
            "目标任务 (Target objective): 在已有磁序附近扩展方向和模长自由度。",
            "建议添加条件 (Add-it trigger): 关注磁各向异性或自旋动力学相关任务。",
            "不建议添加条件 (Avoid trigger): 非磁体系或无磁矩训练目标。",
        ],
        "prereq": [
            "输入结构需包含可用初始磁矩。",
            "先从小角度 `max_angle` 开始。",
        ],
        "fail": [
            "磁矩异常跳变：降低 `max_angle` 与 `magnitude_factor` 范围。",
            "样本膨胀：保持 `num_structures`，优先调幅度而非数量。",
        ],
    },
    "magnetic-order-card.md": {
        "does": "生成 FM/AFM/PM 初始磁序（magnetic order initialization），用于构建多磁态训练数据基础集。",
        "when": [
            "数据症状 (Dataset symptom): 模型只会单一磁序，跨磁序泛化差。",
            "目标任务 (Target objective): 系统覆盖 FM/AFM/PM 分支。",
            "建议添加条件 (Add-it trigger): 研究磁性材料且需多磁序联合训练。",
            "不建议添加条件 (Avoid trigger): 非磁任务或固定单一磁序。",
        ],
        "prereq": [
            "先定义 `magmom_map` 或合理 `default_moment`。",
            "AFM group 模式需先有 Group Label 结果。",
        ],
        "fail": [
            "AFM 分配异常：检查 `afm_mode`、`afm_kvec`、group 标签一致性。",
            "PM 分布偏置：调整 `pm_direction` 与 `pm_balanced`。",
        ],
    },
    "organic-mol-config-pbc-card.md": {
        "does": "对有机体系进行扭转+局域扰动采样，并用键/非键约束控制构象可用性。",
        "when": [
            "数据症状 (Dataset symptom): 分子构象覆盖不足，模型对构象变化敏感。",
            "目标任务 (Target objective): 在保持化学拓扑合理的前提下扩展构象空间。",
            "建议添加条件 (Add-it trigger): 有机晶体、分子体系、多构象任务。",
            "不建议添加条件 (Avoid trigger): 纯无机体系。",
        ],
        "prereq": [
            "确认拓扑可识别，约束参数先用保守值。",
            "先小批量验证有效构象率。",
        ],
        "fail": [
            "有效样本率低：逐步放宽 `bond_keep` 和 `nonbond` 约束。",
            "结构失真：降低 `torsion_range_deg` 与 `gaussian_sigma`。",
        ],
    },
    "perturb-card.md": {
        "does": "对原子坐标施加随机扰动（atomic perturbation），补充近平衡态局部位移样本。",
        "when": [
            "数据症状 (Dataset symptom): 力预测在小位移区间不稳定。",
            "目标任务 (Target objective): 覆盖局域势能面邻域。",
            "建议添加条件 (Add-it trigger): 缺少热噪声近似样本。",
            "不建议添加条件 (Avoid trigger): 已有大量高质量 MD 热扰动数据。",
        ],
        "prereq": [
            "结构应先弛豫到合理状态。",
            "按元素扰动前确认 `element_scalings` 完整。",
        ],
        "fail": [
            "出现短键：降低 `scaling_condition` 或启用元素缩放。",
            "重复样本多：下调 `num_condition` 并加过滤。",
        ],
    },
    "random-doping-card.md": {
        "does": "依据规则表执行替位掺杂（substitutional doping），可选随机采样或比例精确分配。",
        "when": [
            "数据症状 (Dataset symptom): 掺杂浓度和成分覆盖不足。",
            "目标任务 (Target objective): 构建可控掺杂比例和位点分布样本。",
            "建议添加条件 (Add-it trigger): 已明确 target 元素与 dopant 组合。",
            "不建议添加条件 (Avoid trigger): 只需全局占位随机化。",
        ],
        "prereq": [
            "至少配置一条可解析规则。",
            "先用窄浓度区间做正确性验证。",
        ],
        "fail": [
            "规则未生效：检查 `target/dopants/use` 字段。",
            "比例偏差：切换 `doping_type=Exact` 并收窄区间。",
        ],
    },
    "random-occupancy-card.md": {
        "does": "在给定总成分约束下随机分配位点元素（site occupancy assignment），用于同成分多排布样本扩展。",
        "when": [
            "数据症状 (Dataset symptom): 同成分下占位排列单一，迁移泛化差。",
            "目标任务 (Target objective): 增加位点排列多样性而保持总体成分。",
            "建议添加条件 (Add-it trigger): 高熵或多元固溶体占位采样任务。",
            "不建议添加条件 (Avoid trigger): 不需要占位随机化。",
        ],
        "prereq": [
            "确认 `source` 与成分字符串格式。",
            "若使用 group 过滤，结构需已有 group 数组。",
        ],
        "fail": [
            "缺少成分来源：检查 `source/manual`。",
            "统计偏差大：提高 `samples` 或切换 `mode`。",
        ],
    },
    "random-slab-card.md": {
        "does": "按 Miller 指数、层数和真空范围随机生成 slab，构建表面取向与厚度覆盖样本。",
        "when": [
            "数据症状 (Dataset symptom): 表面相关任务误差显著高于体相。",
            "目标任务 (Target objective): 扩展表面几何和边界条件分布。",
            "建议添加条件 (Add-it trigger): 吸附、表面反应、界面任务。",
            "不建议添加条件 (Avoid trigger): 只做体相性质训练。",
        ],
        "prereq": [
            "先用窄 h/k/l 范围试跑。",
            "保证真空层下限足够避免镜像相互作用。",
        ],
        "fail": [
            "slab 过薄：提升 `layer_range` 下限。",
            "表面伪相互作用：提高 `vacuum_range`。",
        ],
    },
    "random-vacancy-card.md": {
        "does": "根据规则删除指定元素原子（rule-based vacancy），控制空位元素类型、数量和区域。",
        "when": [
            "数据症状 (Dataset symptom): 空位缺陷覆盖不足或分布不可控。",
            "目标任务 (Target objective): 精确控制空位类型与局域分布。",
            "建议添加条件 (Add-it trigger): 需要按元素和 group 定向删原子。",
            "不建议添加条件 (Avoid trigger): 仅需无规则随机空位。",
        ],
        "prereq": [
            "先单规则验证，再叠加多规则。",
            "使用 group 时确认输入包含 group 数组。",
        ],
        "fail": [
            "删除总是 0：检查 `element` 和 `count`。",
            "结构过度破坏：收窄规则计数范围。",
        ],
    },
    "shear-angle-card.md": {
        "does": "在保持晶格长度下扰动 alpha/beta/gamma 角，采样角度剪切自由度。",
        "when": [
            "数据症状 (Dataset symptom): 角度相关响应误差高，低对称体系泛化差。",
            "目标任务 (Target objective): 独立覆盖角度畸变通道。",
            "建议添加条件 (Add-it trigger): 研究角度剪切或低对称晶胞变化。",
            "不建议添加条件 (Avoid trigger): 仅关心体积和轴向拉伸。",
        ],
        "prereq": [
            "先小角度范围验证稳定性。",
            "若输入含有机分子，必须开启 `organic`；仅纯无机体系可关闭。",
        ],
        "fail": [
            "晶胞近奇异：降低角度极值。",
            "样本过多：减少三角同时扫描范围。",
        ],
    },
    "shear-matrix-card.md": {
        "does": "通过 xy/yz/xz 剪切矩阵生成非对角形变样本，覆盖剪切应变相关结构变化。",
        "when": [
            "数据症状 (Dataset symptom): 对剪切应力相关性质预测不稳。",
            "目标任务 (Target objective): 系统覆盖剪切分量及对称性差异。",
            "建议添加条件 (Add-it trigger): 需要非对角应变采样。",
            "不建议添加条件 (Avoid trigger): 仅做体积或单轴应变。",
        ],
        "prereq": [
            "先确认 `symmetric` 策略。",
            "单分量试跑后再三分量联扫。",
        ],
        "fail": [
            "形变后结构失真：收窄剪切幅度并优先对称剪切。",
            "组合数过大：增大步长。",
        ],
    },
    "stacking-fault-card.md": {
        "does": "沿指定晶面与滑移步长生成层错样本（stacking fault），补充位错相关局域构型。",
        "when": [
            "数据症状 (Dataset symptom): 层错能或滑移路径预测误差大。",
            "目标任务 (Target objective): 覆盖层错与滑移位移通道。",
            "建议添加条件 (Add-it trigger): 缺陷力学/塑性相关任务。",
            "不建议添加条件 (Avoid trigger): 与位错缺陷无关任务。",
        ],
        "prereq": [
            "先选低指数 `hkl` 平面验证。",
            "先小 `step` 试跑再扩展。",
        ],
        "fail": [
            "滑移方向异常：检查 `hkl` 与晶胞取向。",
            "结构突变过大：降低 `step` 上限。",
        ],
    },
    "super-cell-card.md": {
        "does": "按倍率、目标胞长或原子数上限扩胞（supercell expansion），为缺陷/表面/磁操作提供空间。",
        "when": [
            "数据症状 (Dataset symptom): 原胞太小，周期镜像效应干扰明显。",
            "目标任务 (Target objective): 降低边界伪相互作用并支持复杂操作。",
            "建议添加条件 (Add-it trigger): 下游需要 vacancy/interstitial/slab/magnetic 采样。",
            "不建议添加条件 (Avoid trigger): 算力受限且小胞已满足任务需求。",
        ],
        "prereq": [
            "先选定一种扩胞模式作为主路径。",
            "设置原子数上限避免超预算。",
        ],
        "fail": [
            "结构规模过大：启用 max-atoms 限制。",
            "目标尺寸不达标：检查模式开关与参数对应关系。",
        ],
    },
    "vacancy-defect-card.md": {
        "does": "按数量或浓度随机生成空位缺陷（vacancy sampling），快速覆盖缺陷强度分布。",
        "when": [
            "数据症状 (Dataset symptom): 缺陷密度维度不足，模型对空位数敏感。",
            "目标任务 (Target objective): 快速构建低-中-高缺陷强度样本。",
            "建议添加条件 (Add-it trigger): 需要高通量空位数据且不需复杂规则。",
            "不建议添加条件 (Avoid trigger): 需要按元素/group 精细控制空位位置。",
        ],
        "prereq": [
            "先选 count 或 concentration 单一主模式。",
            "控制 `max_atoms_condition` 先小后大。",
        ],
        "fail": [
            "强度超预期：检查模式开关与参数冲突。",
            "重复度高：调整引擎或 seed 策略。",
        ],
    },
    "vibration-perturb-card.md": {
        "does": "沿振动模方向施加位移扰动（vibrational mode perturbation），比纯随机扰动更贴近动力学自由度。",
        "when": [
            "数据症状 (Dataset symptom): 纯随机扰动不足以覆盖模态方向。",
            "目标任务 (Target objective): 强化声子/振动相关结构覆盖。",
            "建议添加条件 (Add-it trigger): 需要模态驱动的位移样本。",
            "不建议添加条件 (Avoid trigger): 缺少可信振动模式输入。",
        ],
        "prereq": [
            "确认模态输入质量和单位一致。",
            "先小 `amplitude` + 低 `modes_per_sample` 验证。",
        ],
        "fail": [
            "软模导致异常位移：提高 `min_frequency` 或排除近零频。",
            "样本差异不足：先增加模态数，再考虑增大幅度。",
        ],
    },
}


def ensure_profile(name: str) -> dict[str, list[str] | str]:
    if name in CARD_PROFILE:
        return CARD_PROFILE[name]
    title = name.replace(".md", "").replace("-", " ")
    return {
        "does": f"该卡片用于执行 {title} 相关的数据变换与采样，并输出可用于训练的结构变体。",
        "when": [
            "数据症状 (Dataset symptom): 当前数据在该自由度上的覆盖不足。",
            "目标任务 (Target objective): 通过该卡片补齐对应结构变化分布。",
            "建议添加条件 (Add-it trigger): 误差分析已指向该自由度缺口。",
            "不建议添加条件 (Avoid trigger): 当前任务不需要该类变化。",
        ],
        "prereq": [
            "先小规模验证参数效果。",
            "确认该卡片在流程中的顺序正确。",
        ],
        "fail": [
            "结果偏离预期：回退到保守参数逐项放宽。",
            "样本规模失控：加输出上限或下游过滤。",
        ],
    }


def triplet(cons: str, bal: str, exp: str) -> str:
    return "\n".join(
        [
            f"保守：{cons}",
            f"平衡：{bal}",
            f"探索：{exp}",
        ]
    )


ENUM_LIKE_KEYS: set[str] = {
    "mode",
    "engine_type",
    "method",
    "budget_mode",
    "source",
    "distribution",
    "doping_type",
    "afm_mode",
    "pm_direction",
    "axis",
    "kvec",
    "lattice",
    "super_cell_type",
    "preset_index",
    "apply_mode",
    "rotation_mode",
    "plane",
    "slab_mode",
    "pm_mode",
}

PATH_LIKE_KEYS: set[str] = {
    "nep_path",
    "model_path",
    "path",
}

NOTE_STYLE_KEYS_BY_CARD: dict[str, set[str]] = {
    "group-label-card.md": {"mode", "kvec"},
}

BOOL_NOTE_OVERRIDE: dict[str, tuple[str, str]] = {
    "use_seed": ("需要可复现对比时开启。", "探索阶段可关闭以增加随机覆盖。"),
    "overwrite": ("确认允许覆盖已有标签时开启。", "保留已有标签时关闭。"),
    "symmetric": ("需要对称剪切路径时开启。", "仅测试非对称分量时关闭。"),
    "organic": ("输入包含有机分子时必须开启；会先识别团簇并按分子刚性整体移动。", "仅在确认为纯无机体系时关闭。"),
}

SINGLE_NOTE_OVERRIDE: dict[str, str] = {
    "nep_path": "用于生成描述符，默认使用 `src/NepTrainKit/Config/nep89.txt`，可替换为你自己的模型路径。",
    "rules": "按规则语法填写，建议先单规则单帧验证后再扩展。",
    "replacements": "按映射语法填写，建议先用简单映射验证后再扩展。",
    "condition": "按表达式语法填写，建议先用 `all` 验证路径再收紧条件。",
    "group_a": "按项目命名规范填写，需与下游引用保持一致。",
    "group_b": "按项目命名规范填写，需与下游引用保持一致。",
    "card_list": "建议只放共享同一输入且互不依赖的分支卡片；若存在严格前后依赖，请移到组外顺序执行。",
    "filter_card": "用于对组内汇总结果做可选筛选；当前不作为下游卡片输入源。若下游需要过滤结果，请在 Group 后串接独立过滤卡。",
    "params": "按表达式参数表填写，建议先小范围试跑确认语义。",
    "dz_expr": "按表达式语法填写，可替换为自定义位移函数。",
}

TYPE_RANGE_OVERRIDE: dict[tuple[str, str], str] = {
    ("cell-scaling-card.md", "engine_type"): "enum(int): `0=Sobol`, `1=Uniform`",
    ("perturb-card.md", "engine_type"): "enum(int): `0=Sobol`, `1=Uniform`",
    ("vacancy-defect-card.md", "engine_type"): "enum(int): `0=Sobol`, `1=Uniform`",
    ("cell-strain-card.md", "engine_type"): "enum(string): `uniaxial`, `biaxial`, `triaxial`, `isotropic`",
    ("composition-sweep-card.md", "method"): "enum(string): `Grid`, `Sobol`",
    ("interstitial-adsorbate-card.md", "mode"): "enum(int): `0=Interstitial`, `1=Adsorption`（UI 下拉显示字符串）",
    ("interstitial-adsorbate-card.md", "axis"): "enum(int): `0=a(x)`, `1=b(y)`, `2=c(z)`（仅 Adsorption 模式使用）",
}

CARD_UI_LABEL_OVERRIDE: dict[tuple[str, str], str] = {
    ("interstitial-adsorbate-card.md", "species"): "Species comma-separated",
    ("interstitial-adsorbate-card.md", "insert_count"): "Atoms per structure",
    ("interstitial-adsorbate-card.md", "structure_count"): "Structures to generate",
    ("interstitial-adsorbate-card.md", "min_distance"): "Min distance Å",
    ("interstitial-adsorbate-card.md", "use_seed"): "Use seed",
    ("interstitial-adsorbate-card.md", "axis"): "Surface axis",
    ("interstitial-adsorbate-card.md", "offset"): "Offset distance Å",
}

CARD_UI_CAPTION_OVERRIDE: dict[tuple[str, str], str] = {
    ("interstitial-adsorbate-card.md", "mode"): "Mode",
    ("interstitial-adsorbate-card.md", "species"): "Species (comma separated)",
    ("interstitial-adsorbate-card.md", "insert_count"): "Atoms per structure",
    ("interstitial-adsorbate-card.md", "structure_count"): "Structures to generate",
    ("interstitial-adsorbate-card.md", "min_distance"): "Min distance (Å)",
    ("interstitial-adsorbate-card.md", "max_attempts"): "Max attempts",
    ("interstitial-adsorbate-card.md", "use_seed"): "Use seed",
    ("interstitial-adsorbate-card.md", "seed"): "Seed",
    ("interstitial-adsorbate-card.md", "axis"): "Surface axis",
    ("interstitial-adsorbate-card.md", "offset"): "Offset distance (Å)",
}

CARD_SET_RANGE_HINT: dict[tuple[str, str], str] = {
    ("interstitial-adsorbate-card.md", "insert_count"): "1-20",
    ("interstitial-adsorbate-card.md", "structure_count"): "1-1000",
    ("interstitial-adsorbate-card.md", "min_distance"): "0.0-10.0",
    ("interstitial-adsorbate-card.md", "max_attempts"): "1-1000",
    ("interstitial-adsorbate-card.md", "seed"): f"0-{2**31 - 1}",
    ("interstitial-adsorbate-card.md", "offset"): "0.0-10.0",
}


def display_type_range(
    card_file: str,
    key: str,
    typ: str,
    runtime_defaults: dict[str, object],
) -> str:
    override = TYPE_RANGE_OVERRIDE.get((card_file, key))
    if override:
        return override
    if key in runtime_defaults:
        value = runtime_defaults[key]
        if isinstance(value, bool):
            return "bool"
        if key in ENUM_LIKE_KEYS:
            if isinstance(value, int):
                return "enum(int)"
            if isinstance(value, str):
                return "enum(string)"
        if isinstance(value, list) and len(value) == 1:
            single = value[0]
            if isinstance(single, bool):
                return "bool（单值输入）"
            if isinstance(single, int):
                return "int（单值输入）"
            if isinstance(single, float):
                return "float（单值输入）"
            if isinstance(single, str):
                return "string（单值输入）"
    return typ


def recommendation_style(card_file: str, key: str, typ: str) -> str:
    key_l = key.strip().lower()
    typ_l = typ.strip().lower()
    if key_l in NOTE_STYLE_KEYS_BY_CARD.get(card_file, set()):
        return "note"
    if key_l in PATH_LIKE_KEYS or key_l.endswith("_path") or "path" in key_l:
        return "note"
    if "bool" in typ_l:
        return "binary"
    if key_l in ENUM_LIKE_KEYS:
        return "tiered"
    if "string" in typ_l:
        return "note"
    if any(token in typ_l for token in ("int", "float", "list[")):
        return "tiered"
    return "note"


def bool_note(key: str, ui: str) -> tuple[str, str]:
    if key in BOOL_NOTE_OVERRIDE:
        return BOOL_NOTE_OVERRIDE[key]
    return (
        f"需要启用 `{ui}` 对应行为时开启。",
        f"希望保持默认/更保守行为时关闭。",
    )


def single_note(key: str, ui: str, typ: str) -> str:
    if key in SINGLE_NOTE_OVERRIDE:
        return SINGLE_NOTE_OVERRIDE[key]
    if "string" in typ.lower():
        return f"`{ui}` 可按任务替换为自定义值；建议先用最小样本验证后再批量生成。"
    return "建议围绕任务目标小步调整，并先做单帧验证。"


def widget_hint(typ: str) -> str:
    typ_l = typ.lower()
    if "enum(" in typ_l:
        return "下拉选择 `ComboBox`（显示文本与序列化值可能不同）。"
    if "bool" in typ_l:
        return "勾选开关 `CheckBox`。"
    if "list[3]" in typ_l:
        return "区间输入 `SpinBoxUnitInputFrame`（`min/max/step` 三输入框）。"
    if "单值输入" in typ or "int" in typ_l or "float" in typ_l:
        return "数值输入 `SpinBoxUnitInputFrame`。"
    if "string" in typ_l:
        return "文本输入 `LineEdit`（或可编辑下拉）。"
    return "按字段类型解析。"


def is_numeric_tiered_type(typ: str) -> bool:
    typ_l = typ.lower()
    if "enum(" in typ_l or "bool" in typ_l or "string" in typ_l:
        return False
    return any(token in typ_l for token in ("int", "float", "list["))


def recommendation_has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text))


def recommendation_tiers_have_numbers(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if "：" in ln]
    if not lines:
        return False
    for ln in lines:
        right = ln.split("：", 1)[1]
        if not re.search(r"\d", right):
            return False
    return True


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3g}"


def _single_numeric_default(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
        return float(value[0])
    return None


def auto_numeric_triplet(key: str, value: object) -> str:
    key_l = key.lower()
    if "seed" in key_l:
        return triplet("0（随机）", "1-99（可复现）", "100-9999（多 seed 对比）")
    if key_l == "rep":
        return triplet("1x1x1 到 2x2x2", "2x2x2 到 3x3x3", "3x3x3 到 5x5x5")
    if key_l in {"hkl", "h_range", "k_range", "l_range"}:
        return triplet("低指数（1-2）", "中指数（2-4）", "高指数（4-6）")
    v = _single_numeric_default(value)
    if v is None:
        if isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(x, (int, float)) for x in value):
            a, b, s = float(value[0]), float(value[1]), float(value[2])
            s_mid = max(1e-6, s)
            return triplet(
                f"{_fmt_num(a)} 到 {_fmt_num(b)}，step {_fmt_num(s_mid)}",
                f"{_fmt_num(a)} 到 {_fmt_num(b)}，step {_fmt_num(max(1e-6, s_mid / 2))}",
                f"{_fmt_num(a)} 到 {_fmt_num(b)}，step {_fmt_num(s_mid * 2)}",
            )
        return triplet("1-2", "2-5", "5-10")
    if "min_distance" in key_l or (key_l.startswith("min_") and "distance" in key_l):
        c_low = max(0.05, 1.2 * v)
        c_high = max(c_low + 0.01, 1.8 * v)
        b_low = max(0.05, 0.9 * v)
        b_high = max(b_low + 0.01, 1.2 * v)
        e_low = max(0.05, 0.6 * v)
        e_high = max(e_low + 0.01, 0.9 * v)
        return triplet(
            f"{_fmt_num(c_low)}-{_fmt_num(c_high)}",
            f"{_fmt_num(b_low)}-{_fmt_num(b_high)}",
            f"{_fmt_num(e_low)}-{_fmt_num(e_high)}（仅探索）",
        )

    if abs(v - round(v)) < 1e-9 and v >= 1:
        vi = int(round(v))
        c_low, c_high = max(1, int(round(0.5 * vi))), max(1, vi)
        b_low, b_high = max(1, vi), max(2, int(round(2.0 * vi)))
        e_low, e_high = max(2, int(round(2.0 * vi))), max(4, int(round(5.0 * vi)))
        return triplet(
            f"{c_low}-{c_high}",
            f"{b_low}-{b_high}",
            f"{e_low}-{e_high}",
        )

    if v <= 0:
        return triplet("0.1-0.5", "0.5-1.5", "1.5-3.0")

    c_low, c_high = 0.7 * v, 1.0 * v
    b_low, b_high = 1.0 * v, 1.5 * v
    e_low, e_high = 1.5 * v, 2.5 * v
    return triplet(
        f"{_fmt_num(c_low)}-{_fmt_num(c_high)}",
        f"{_fmt_num(b_low)}-{_fmt_num(b_high)}",
        f"{_fmt_num(e_low)}-{_fmt_num(e_high)}",
    )


GENERIC_KEY_INFO: dict[str, tuple[str, str, str]] = {
    "use_seed": (
        "是否启用固定随机种子 (deterministic seed switch)。",
        "开启后可复现实验；关闭后每次采样分布会变化。",
        triplet("探索阶段关闭", "对比实验开启", "复现实验必须开启"),
    ),
    "seed": (
        "随机种子值 (random seed value)。",
        "只影响随机路径，不改变物理模型本身。",
        triplet("0 或空表示随机", "固定整数便于复现", "多 seed 扫描评估不确定性"),
    ),
    "rules": (
        "规则列表输入 (rule list)。",
        "决定对象筛选和操作强度，是规则型卡片核心参数。",
        triplet("单规则先跑通", "多规则渐进扩展", "复杂规则并行需审计"),
    ),
    "num_condition": (
        "采样数量控制 (sample count control)。",
        "主要影响输出规模与耗时，不是幅度主控参数。",
        triplet("小批量验证", "中等规模训练", "大规模需过滤"),
    ),
    "max_outputs": (
        "输出上限 (maximum outputs)。",
        "限制样本规模，防止组合爆炸。",
        triplet("先设小上限", "按预算放宽", "探索时提高并加过滤"),
    ),
    "max_atoms_condition": (
        "每帧最大生成数 (max generated structures per frame)。",
        "主要控制数据量和运行时间。",
        triplet("10-50", "50-200", "200+ 需 FPS"),
    ),
    "scaling_condition": (
        "扰动幅度参数 (perturbation amplitude)。",
        "是形变/位移强度主控量。",
        triplet("小幅度", "中幅度", "大幅度需质量筛选"),
    ),
    "mode": (
        "操作模式 (operation mode)。",
        "改变执行逻辑路径，影响样本分布。",
        triplet("默认模式先验证", "按任务切换", "探索模式配审计"),
    ),
    "organic": (
        "有机团簇识别与刚性移动开关 (organic cluster rigid mode)。",
        "开启后先识别有机团簇，扰动时对有机分子做刚性整体移动，减少分子内键长/拓扑被破坏；输入含有机分子时应开启。",
        triplet("无机关闭", "混合体系按需", "有机体系开启"),
    ),
    "elements": (
        "元素集合输入 (element set)。",
        "决定参与操作的元素子集。",
        triplet("核心元素", "核心+次要元素", "全元素覆盖需审计"),
    ),
    "engine_type": (
        "算法引擎/类型 (engine type)。",
        "不同引擎影响采样均匀性和速度。",
        triplet("默认引擎", "双引擎对比", "按覆盖目标选择"),
    ),
    "perturb_angle": (
        "角度扰动开关 (perturb angle switch)。",
        "开启后除尺度变化外还会引入晶格角度变化，覆盖更广但失稳风险更高。",
        triplet("默认关闭", "角度敏感任务开启", "大角度探索需后筛"),
    ),
    "method": (
        "候选生成方法 (sampling method)。",
        "改变候选点分布形状与覆盖均匀性。",
        triplet("可解释方法", "均匀覆盖方法", "探索方法"),
    ),
    "order": (
        "组合阶数范围 (order range)。",
        "阶数越高，候选组合数增长越快。",
        triplet("低阶", "中阶", "高阶需强预算"),
    ),
    "n_points": (
        "采样点数 (number of points)。",
        "点数越大覆盖越密，但计算开销更高。",
        triplet("低密度验证", "中密度训练", "高密度配过滤"),
    ),
    "min_fraction": (
        "最小组分占比 (minimum fraction)。",
        "抬高下限会过滤极端稀释成分。",
        triplet("较高下限更稳", "中等下限平衡", "低下限覆盖边角"),
    ),
    "include_endpoints": (
        "是否包含端点 (include endpoints)。",
        "控制是否保留纯端元和边界成分点。",
        triplet("包含端点", "按任务裁剪", "只采中区间"),
    ),
    "budget_mode": (
        "预算分配策略 (budget mode)。",
        "决定不同子空间获得样本名额的比例。",
        triplet("均匀分配", "轻度偏置", "强偏置探索"),
    ),
    "target": (
        "目标元素 (target species)。",
        "限定被替换或处理的原子种类。",
        triplet("单 target", "分批多 target", "并行多 target 建议拆卡"),
    ),
    "replacements": (
        "替换映射 (replacement mapping)。",
        "决定替换后元素组合与比例。",
        triplet("单元素", "双元素比例", "多元素需后筛"),
    ),
    "condition": (
        "条件表达式 (condition expression)。",
        "控制筛选区域与命中数量。",
        triplet("窄条件", "中范围", "全局条件仅探索"),
    ),
    "lattice": (
        "晶型模板 (lattice prototype)。",
        "决定生成结构的拓扑基底。",
        triplet("主晶型", "主+次晶型", "全晶型需预算限制"),
    ),
    "element": (
        "元素类型 (element type)。",
        "定义当前操作核心元素。",
        triplet("核心元素", "主次元素", "全元素需审计"),
    ),
    "a_range": (
        "晶格常数范围 (lattice constant range)。",
        "控制结构尺寸扫描范围。",
        triplet("窄范围", "中范围", "宽范围需稳定性检查"),
    ),
    "covera": (
        "c/a 比例 (c over a ratio)。",
        "影响非立方晶型的几何各向异性。",
        triplet("文献值附近", "小范围扫描", "大偏移仅探索"),
    ),
    "auto_supercell": (
        "自动扩胞开关 (auto supercell)。",
        "自动根据目标规模调整复制参数。",
        triplet("默认开启", "按需切换", "关闭时需手动控规模"),
    ),
    "max_atoms": (
        "最大原子数 (maximum atoms)。",
        "限制单结构规模，避免超算力预算。",
        triplet("中小规模", "预算内放宽", "超大规模仅少量样本"),
    ),
    "rep": (
        "复制倍率向量 (replication vector)。",
        "直接决定扩胞倍数和原子数增长。",
        triplet("2-3 倍", "3-5 倍", "更高倍需强约束"),
    ),
    "nep_path": (
        "特征模型路径 (NEP model path)。",
        "用于距离特征计算，路径失效会导致过滤退化。",
        triplet("已验证模型", "版本化管理", "多模型对比逐一校验"),
    ),
    "min_distance_condition": (
        "最小特征距离阈值 (minimum descriptor distance)。",
        "阈值越大去冗余越强，保留样本越少。",
        triplet("低阈值保覆盖", "中阈值平衡", "高阈值强压缩"),
    ),
    "kvec": (
        "k 向量规则 (k-vector rule)。",
        "影响子晶格分组结果和后续磁序构造。",
        triplet("简单晶向", "主晶向覆盖", "复杂晶向先可视化"),
    ),
    "group_a": (
        "A 组标签 (group A label)。",
        "定义分组命名供下游引用。",
        triplet("统一命名", "按项目命名", "复杂命名需文档化"),
    ),
    "group_b": (
        "B 组标签 (group B label)。",
        "定义分组命名供下游引用。",
        triplet("统一命名", "按项目命名", "复杂命名需文档化"),
    ),
    "overwrite": (
        "覆盖已有标签 (overwrite existing labels)。",
        "开启会重写旧 group 信息。",
        triplet("默认关闭保守", "确认规则后开启", "多流程并行谨慎开启"),
    ),
    "species": (
        "插入物种 (inserted species)。",
        "定义插入元素或片段类型。",
        triplet("单物种", "主+次物种", "多物种分批验证"),
    ),
    "insert_count": (
        "每次插入数量 (insert count)。",
        "提高会增强缺陷强度并增加碰撞风险。",
        triplet("1", "2-3", "4+ 需高尝试次数"),
    ),
    "structure_count": (
        "每帧生成结构数 (structures per frame)。",
        "主要影响数据规模和运行时。",
        triplet("10 左右", "20-100", "200+ 需过滤"),
    ),
    "min_distance": (
        "最小原子间距阈值 (minimum allowed distance)。",
        "越大越保守但成功率会下降。",
        triplet("高阈值保物理性", "中阈值平衡", "低阈值仅探索"),
    ),
    "max_attempts": (
        "最大尝试次数 (maximum attempts)。",
        "提高成功率但会增加耗时。",
        triplet("10-50", "100-300", "500+ 仅复杂体系"),
    ),
    "axis": (
        "作用轴/方向 (axis)。",
        "改变操作方向定义，直接影响输出分布。",
        triplet("标准轴", "按主晶向设置", "自定义轴先可视化"),
    ),
    "offset": (
        "偏移参数 (offset)。",
        "控制相对参考面或方向的位移量。",
        triplet("小偏移", "中偏移", "大偏移仅探索"),
    ),
    "preset_index": (
        "预设索引 (preset index)。",
        "选择内置变换模板。",
        triplet("默认预设", "按体系切预设", "复杂预设先单帧验证"),
    ),
    "dz_expr": (
        "位移表达式 (displacement expression)。",
        "决定空间相关层位移函数形式。",
        triplet("简单函数", "复合函数", "高阶函数需单帧审计"),
    ),
    "params": (
        "表达式参数 (expression parameters)。",
        "用于调节 `dz_expr` 形状和幅度。",
        triplet("少参数", "2-3 参数联调", "多参数需网格搜索"),
    ),
    "apply_mode": (
        "应用模式 (apply mode)。",
        "决定操作作用对象和范围。",
        triplet("局部模式", "分层模式", "全局模式仅探索"),
    ),
    "z_range": (
        "Z 向范围 (z range)。",
        "控制沿 z 方向的作用区间或幅度。",
        triplet("窄范围", "中范围", "宽范围需防碰撞"),
    ),
    "wrap": (
        "回卷边界 (wrap to cell)。",
        "开启后坐标会映射回周期胞内。",
        triplet("周期体系开启", "混合体系按需", "非周期可关闭"),
    ),
    "extend_cell_z": (
        "扩展 z 晶胞 (extend cell in z)。",
        "用于避免层复制后跨边界冲突。",
        triplet("默认开启", "特殊导出再关", "关闭时手动查边界"),
    ),
    "extra_vacuum": (
        "额外真空层 (extra vacuum)。",
        "增大可降低镜像相互作用。",
        triplet("0-5Å", "5-15Å", "20Å+ 强隔离"),
    ),
    "layers": (
        "层参数 (layer index/count)。",
        "控制操作层位或层数覆盖。",
        triplet("浅层", "中层", "深层需稳定性审计"),
    ),
    "distance": (
        "距离参数 (distance parameter)。",
        "过小会导致碰撞，过大会稀释作用强度。",
        triplet("偏大保守", "中等平衡", "偏小仅探索"),
    ),
    "max_angle": (
        "最大角度参数 (maximum angle)。",
        "主控角度扰动强度。",
        triplet("小角度", "中角度", "大角度需筛选"),
    ),
    "num_structures": (
        "每帧输出结构数 (structures per frame)。",
        "影响数据体量，不直接决定单样本幅度。",
        triplet("5-10", "10-30", "30+ 配过滤"),
    ),
    "lift_scalar": (
        "标量提升开关 (lift scalar)。",
        "控制标量输入是否映射到向量表示。",
        triplet("默认开启", "按输入格式调整", "关闭需保证向量输入"),
    ),
    "disturb_magnitude": (
        "模长扰动开关 (disturb magnitude)。",
        "开启后会拓宽磁矩长度分布。",
        triplet("先关闭只转角", "开启小范围", "大范围需后筛"),
    ),
    "magnitude_factor": (
        "模长缩放区间 (magnitude factor range)。",
        "决定磁矩长度扰动幅度。",
        triplet("窄区间", "中区间", "宽区间需筛选"),
    ),
    "format": (
        "磁矩格式 (magmom format)。",
        "决定共线标量还是非共线向量表示。",
        triplet("共线基线", "按任务切向量", "混合格式需校验"),
    ),
    "magmom_map": (
        "元素磁矩映射 (element moment map)。",
        "定义元素到初始磁矩的映射关系。",
        triplet("核心磁元素先配", "补齐全元素", "复杂映射分批验证"),
    ),
    "use_element_dirs": (
        "元素方向模板开关 (use element directions)。",
        "允许不同元素采用不同方向先验。",
        triplet("默认关闭", "异元素方向需求时开启", "开启后逐元素核查"),
    ),
    "default_moment": (
        "默认磁矩 (default moment)。",
        "未命中映射元素时的后备值。",
        triplet("小默认值", "经验值", "大默认值仅探索"),
    ),
    "apply_elements": (
        "应用元素列表 (apply elements)。",
        "限制哪些元素执行当前磁序策略。",
        triplet("仅磁元素", "主次元素", "全元素需审计"),
    ),
    "gen_fm": (
        "生成 FM 分支 (generate FM)。",
        "控制是否输出铁磁样本。",
        triplet("开启", "与 AFM 平衡", "特殊任务可关闭"),
    ),
    "gen_afm": (
        "生成 AFM 分支 (generate AFM)。",
        "控制是否输出反铁磁样本。",
        triplet("按任务开启", "与 FM 联合", "复杂 AFM 需 group 支持"),
    ),
    "afm_mode": (
        "AFM 构造模式 (AFM mode)。",
        "决定用 k-vector 还是 group 构造正负子晶格。",
        triplet("k-vector 先跑通", "按结构切 group", "混合模式需审计"),
    ),
    "afm_kvec": (
        "AFM k 向量 (AFM k-vector)。",
        "控制 AFM 反转周期方向。",
        triplet("简单晶向", "主晶向覆盖", "复杂方向先验证"),
    ),
    "afm_group_a": (
        "AFM A 组标签 (AFM group A)。",
        "group 模式下正向子晶格标签。",
        triplet("标准标签", "与 Group Label 对齐", "复杂命名需文档化"),
    ),
    "afm_group_b": (
        "AFM B 组标签 (AFM group B)。",
        "group 模式下反向子晶格标签。",
        triplet("标准标签", "与 Group Label 对齐", "复杂命名需文档化"),
    ),
    "afm_zero_unknown": (
        "未知元素置零 (zero unknown moments)。",
        "防止未配置元素引入噪声磁矩。",
        triplet("默认开启", "映射完整后可关闭", "关闭时必须全映射"),
    ),
    "gen_pm": (
        "生成 PM 分支 (generate PM)。",
        "控制是否输出顺磁样本。",
        triplet("少量开启", "与 FM/AFM 平衡", "仅 PM 任务大量开启"),
    ),
    "pm_count": (
        "PM 样本数 (PM sample count)。",
        "控制 PM 分支输出规模。",
        triplet("5-10", "10-30", "30+ 配过滤"),
    ),
    "pm_direction": (
        "PM 方向分布 (PM direction distribution)。",
        "决定顺磁方向采样模式。",
        triplet("sphere", "cone", "定向分布仅专题研究"),
    ),
    "pm_cone_angle": (
        "PM 锥角 (PM cone angle)。",
        "仅在 cone 模式控制偏离主轴幅度。",
        triplet("小锥角", "中锥角", "大锥角接近全向"),
    ),
    "pm_balanced": (
        "PM 平衡开关 (PM balanced switch)。",
        "控制方向采样是否保持正负平衡。",
        triplet("开启保平衡", "按任务切换", "关闭用于偏置研究"),
    ),
    "perturb_per_frame": (
        "每帧扰动数 (perturbations per frame)。",
        "主要影响样本规模和运行时间。",
        triplet("20-50", "50-150", "200+ 配过滤"),
    ),
    "torsion_range_deg": (
        "扭转角范围 (torsion range)。",
        "主控构象变化幅度。",
        triplet("±20~30°", "±45~60°", "±90°+ 仅探索"),
    ),
    "max_torsions_per_conf": (
        "每构象扭转数上限 (max torsions per config)。",
        "越大构象变化越复杂。",
        triplet("1-3", "4-6", "7+"),
    ),
    "gaussian_sigma": (
        "高斯扰动强度 (gaussian sigma)。",
        "主控局域随机位移幅度。",
        triplet("0.01-0.02", "0.03-0.05", "0.08+ 需后筛"),
    ),
    "pbc_mode": (
        "周期处理模式 (PBC mode)。",
        "决定周期/非周期下的约束处理路径。",
        triplet("auto", "按体系手动指定", "混合模式逐项验证"),
    ),
    "local_cutoff": (
        "局域截断半径 (local cutoff)。",
        "影响邻域构造范围与性能。",
        triplet("较小高效", "中等平衡", "较大仅复杂分子"),
    ),
    "local_subtree": (
        "局域子图规模 (local subtree size)。",
        "控制拓扑搜索深度。",
        triplet("小子图", "中子图", "大子图高成本"),
    ),
    "bond_detect_factor": (
        "成键检测因子 (bond detect factor)。",
        "越大越容易判定成键。",
        triplet("保守检测", "平衡检测", "宽松检测需后筛"),
    ),
    "bond_keep_min_factor": (
        "最小保键因子 (bond keep min factor)。",
        "限制最短可接受键长比例。",
        triplet("较大更保守", "中等", "较小更激进"),
    ),
    "bond_keep_max_factor": (
        "最大保键因子 (bond keep max factor)。",
        "限制最长可接受键长比例。",
        triplet("较小更严格", "中等", "较大更宽松"),
    ),
    "bond_keep_max_enable": (
        "启用最大保键约束 (enable max bond keep)。",
        "决定是否执行键长上限约束。",
        triplet("默认关闭", "拉长异常时开启", "全程开启需谨慎"),
    ),
    "nonbond_min_factor": (
        "非键最小距离因子 (nonbond min factor)。",
        "过小会增加非键碰撞风险。",
        triplet("偏大保守", "中等平衡", "偏小仅探索"),
    ),
    "max_retries": (
        "最大重试次数 (max retries)。",
        "提高有效样本率但增加耗时。",
        triplet("5-10", "10-30", "30+"),
    ),
    "mult_bond_factor": (
        "多键修正因子 (multiple-bond factor)。",
        "调节多重键约束强度。",
        triplet("默认附近", "小范围扫描", "大偏移需人工核查"),
    ),
    "nonpbc_box_size": (
        "非周期盒尺寸 (non-PBC box size)。",
        "定义非周期模式的可用空间尺度。",
        triplet("较大盒保守", "按分子尺寸设置", "过小盒仅测试"),
    ),
    "bo_c_const": (
        "键级常数 C (bond-order constant C)。",
        "影响键级衰减曲线形状。",
        triplet("默认附近", "小范围微调", "大范围仅方法研究"),
    ),
    "bo_threshold": (
        "键级阈值 (bond-order threshold)。",
        "控制成键/断键判定边界。",
        triplet("高阈值保守", "中阈值", "低阈值激进"),
    ),
    "use_element_scaling": (
        "按元素扰动缩放 (use element scaling)。",
        "允许不同元素使用不同扰动幅度。",
        triplet("先关闭统一扰动", "关键元素差异化", "全元素差异化需审计"),
    ),
    "element_scalings": (
        "元素缩放字典 (element scaling dict)。",
        "定义元素到扰动系数映射。",
        triplet("主元素先配", "主次元素", "全元素映射+后筛"),
    ),
    "doping_type": (
        "掺杂采样类型 (doping type)。",
        "Random 强随机性，Exact 更接近目标比例。",
        triplet("Exact 基线", "Exact+Random 对比", "Random 探索扩展"),
    ),
    "source": (
        "成分来源 (composition source)。",
        "决定自动读取还是手工输入。",
        triplet("自动优先", "手工兜底", "双来源交叉核验"),
    ),
    "manual": (
        "手动成分字符串 (manual composition)。",
        "用于显式指定元素比例。",
        triplet("简单配方", "主流多元", "高元配方需归一化检查"),
    ),
    "samples": (
        "每帧样本数 (samples per frame)。",
        "控制输出体量和统计稳定性。",
        triplet("1-3", "5-10", "20+ 需去重"),
    ),
    "group_filter": (
        "分组过滤条件 (group filter)。",
        "限制操作仅作用于指定 group。",
        triplet("先不过滤", "主目标组过滤", "多组过滤需覆盖检查"),
    ),
    "h_range": (
        "h 指数范围 (h range)。",
        "控制表面取向扫描维度之一。",
        triplet("窄范围", "中范围", "宽范围需预算限制"),
    ),
    "k_range": (
        "k 指数范围 (k range)。",
        "控制表面取向扫描维度之一。",
        triplet("窄范围", "中范围", "宽范围需预算限制"),
    ),
    "l_range": (
        "l 指数范围 (l range)。",
        "控制表面取向扫描维度之一。",
        triplet("低指数", "中指数", "高指数仅探索"),
    ),
    "layer_range": (
        "层数范围 (layer range)。",
        "主控 slab 厚度分布。",
        triplet("薄层", "中厚", "厚层高成本"),
    ),
    "vacuum_range": (
        "真空层范围 (vacuum range)。",
        "影响表面镜像相互作用强度。",
        triplet("10Å 左右", "12-20Å", "20Å+"),
    ),
    "alpha_range": (
        "alpha 角扫描范围 (alpha range)。",
        "控制 alpha 角扰动幅度。",
        triplet("±1°", "±3°", "±6°"),
    ),
    "beta_range": (
        "beta 角扫描范围 (beta range)。",
        "控制 beta 角扰动幅度。",
        triplet("±1°", "±3°", "±6°"),
    ),
    "gamma_range": (
        "gamma 角扫描范围 (gamma range)。",
        "控制 gamma 角扰动幅度。",
        triplet("±1°", "±3°", "±6°"),
    ),
    "symmetric": (
        "对称剪切开关 (symmetric shear)。",
        "开启后更接近对称形变路径，通常更稳定。",
        triplet("开启", "按任务切换", "关闭仅探索非对称路径"),
    ),
    "xy_range": (
        "xy 剪切范围 (xy shear range)。",
        "控制 xy 分量剪切幅度。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    "yz_range": (
        "yz 剪切范围 (yz shear range)。",
        "控制 yz 分量剪切幅度。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    "xz_range": (
        "xz 剪切范围 (xz shear range)。",
        "控制 xz 分量剪切幅度。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    "hkl": (
        "Miller 指数 (hkl plane)。",
        "定义层错/滑移操作的晶面。",
        triplet("低指数面", "主晶面组合", "高指数面需验证"),
    ),
    "step": (
        "步长区间 (step range)。",
        "主控扫描位移幅度与分辨率。",
        triplet("小步长细扫", "中步长平衡", "大步长仅探索"),
    ),
    "super_cell_type": (
        "超胞模式类型 (supercell mode type)。",
        "决定采用倍率、目标胞长或原子上限策略。",
        triplet("单模式先跑通", "按任务切换", "多模式并行需对照"),
    ),
    "super_scale_radio_button": (
        "倍率模式开关 (scale mode switch)。",
        "控制是否按固定倍率扩胞。",
        triplet("倍率任务开启", "混合任务按需", "非倍率任务关闭"),
    ),
    "super_scale_condition": (
        "倍率参数 (scale factors)。",
        "定义各方向复制倍数。",
        triplet("2x 左右", "2-4x", "5x+ 高成本"),
    ),
    "super_cell_radio_button": (
        "目标胞长模式开关 (target-cell mode switch)。",
        "控制是否按目标胞长扩胞。",
        triplet("尺寸任务开启", "与倍率二选一", "并开时需核优先级"),
    ),
    "super_cell_condition": (
        "目标胞长参数 (target cell condition)。",
        "定义扩胞后的最小胞长目标。",
        triplet("中等胞长", "较大胞长", "超大胞长需预算"),
    ),
    "max_atoms_radio_button": (
        "原子上限模式开关 (max-atoms mode switch)。",
        "用于限制扩胞后结构规模。",
        triplet("预算紧张开启", "常规按需", "预算充足可关闭"),
    ),
    "num_radio_button": (
        "计数模式开关 (count mode switch)。",
        "按绝对数量控制缺陷强度。",
        triplet("小体系常用", "中体系平衡", "大体系注意尺度效应"),
    ),
    "concentration_radio_button": (
        "浓度模式开关 (concentration mode switch)。",
        "按比例控制缺陷强度。",
        triplet("固定浓度任务开启", "与 count 二选一", "高浓度仅探索"),
    ),
    "concentration_condition": (
        "浓度参数 (concentration parameter)。",
        "定义缺陷比例上限。",
        triplet("低浓度", "中浓度", "高浓度需稳定性筛查"),
    ),
    "distribution": (
        "采样分布类型 (distribution type)。",
        "决定随机变量分布形状。",
        triplet("均匀分布", "高斯分布", "重尾分布仅探索"),
    ),
    "amplitude": (
        "振幅参数 (amplitude)。",
        "主控扰动强度。",
        triplet("小振幅", "中振幅", "大振幅需后筛"),
    ),
    "modes_per_sample": (
        "每样本模态数 (modes per sample)。",
        "控制扰动方向组合复杂度。",
        triplet("1-2", "3-4", "5+"),
    ),
    "min_frequency": (
        "最小频率阈值 (minimum frequency)。",
        "过滤过低频模式，减少软模异常。",
        triplet("较高阈值更稳", "中阈值平衡", "低阈值覆盖更广"),
    ),
    "max_num": (
        "每帧最大样本数 (max samples per frame)。",
        "控制输出规模。",
        triplet("10-20", "20-60", "100+"),
    ),
    "scale_by_frequency": (
        "按频率缩放开关 (scale by frequency)。",
        "开启后高频模位移更小，通常更物理。",
        triplet("默认开启", "对比开关影响", "关闭仅方法测试"),
    ),
    "exclude_near_zero": (
        "排除近零频开关 (exclude near-zero)。",
        "减少平移/旋转伪模引起的异常位移。",
        triplet("开启更稳", "按质量切换", "关闭需提高频率阈值"),
    ),
    "card_list": (
        "组内卡片列表 (card list)。",
        "同一输入的分支集合，按布局顺序依次运行并汇总输出。",
        triplet("仅稳定卡片", "渐进扩组", "复杂流程建议分层"),
    ),
    "filter_card": (
        "组内过滤节点 (filter card)。",
        "对组内汇总结果执行可选筛选；当前不作为下游卡片输入源。",
        triplet("先不启过滤", "单过滤节点", "多过滤建议拆组"),
    ),
    "x_range": (
        "X 方向应变范围 (x range)。",
        "定义 x 轴方向应变扫描。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    "y_range": (
        "Y 方向应变范围 (y range)。",
        "定义 y 轴方向应变扫描。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    "order": (
        "组合阶范围 (order)。",
        "控制元数组合阶数。",
        triplet("低阶", "中阶", "高阶需预算"),
    ),
}


CARD_KEY_OVERRIDE: dict[tuple[str, str], tuple[str, str, str]] = {
    ("cell-scaling-card.md", "scaling_condition"): (
        "最大缩放幅度系数 `m` (max scaling ratio)。取值为 `0-1` 的比例值；例如 `0.04` 表示 `4%`。",
        "长度/角度因子按 `1±m` 采样；例如 `m=0.04` 时，晶格长度与角度扰动幅度约为 `±4%`。",
        triplet("0.01-0.03（约 1%-3%）", "0.03-0.06（约 3%-6%）", "0.06-0.1（约 6%-10%，需严格质检）"),
    ),
    ("cell-scaling-card.md", "num_condition"): (
        "每个输入结构的晶格扰动采样数 (samples per input structure)。",
        "该卡片主要改变晶格尺度/角度，原子坐标会随晶格同步缩放；单纯增大该值会快速放大数据量，但微观局域多样性提升有限。细粒度多样性建议主要由 `Atomic Perturb` 等原子级扰动补充。",
        triplet("8-20（先验证分布与质检流程）", "20-50（常规训练覆盖）", "50-100（仅在明确需要更多晶格态时，建议联用 Atomic Perturb）"),
    ),
    ("perturb-card.md", "scaling_condition"): (
        "最大位移距离 (max displacement distance)，单位 `Å`。",
        "这是绝对长度而非百分比；每个原子的位移向量按 `[-1,1]` 随机方向乘以该距离上限（或元素专属上限）。",
        triplet("0.05-0.15 Å", "0.15-0.30 Å", "0.30-0.50 Å（建议配后筛）"),
    ),
    ("vacancy-defect-card.md", "concentration_condition"): (
        "空位浓度比例 (vacancy concentration ratio)。取值 `0-1`，可按百分比理解为 `0%-100%`。",
        "在浓度模式下，最大空位数按 `int(concentration * n_atoms)` 计算；例如 `0.02` 表示约 `2%` 原子被删。",
        triplet("0.005-0.02（约 0.5%-2%）", "0.02-0.08（约 2%-8%）", "0.08-0.20（约 8%-20%，需稳定性评估）"),
    ),
    ("composition-sweep-card.md", "min_fraction"): (
        "最小组分比例下限 (minimum fraction ratio)。取值 `0-1`，可按百分比理解。",
        "约束每个元素的最小占比；例如 `0.05` 表示每个组分至少 `5%`，会过滤极端稀释端点。",
        triplet("0.05-0.10（更稳）", "0.01-0.05（平衡覆盖）", "0.00-0.01（覆盖边角但组合更散）"),
    ),
    ("cell-scaling-card.md", "engine_type"): (
        "随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。",
        "Uniform 生成更快；Sobol 在样本较少时覆盖更均匀。样本规模足够大时，两者分布差异通常会缩小。",
        triplet("小样本优先 Sobol", "先用 Uniform 快速试跑再抽样对比 Sobol", "大样本阶段按速度与复现需求择优"),
    ),
    ("perturb-card.md", "engine_type"): (
        "随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。",
        "Uniform 更快，适合高吞吐批量扰动；Sobol 在少量样本时能更均匀覆盖位移方向。样本数量增大后，两者差距通常减小。",
        triplet("少量样本与基线验证用 Sobol", "中等规模用 Uniform 预跑并抽查 Sobol", "大规模以吞吐优先，可固定一种引擎保持一致性"),
    ),
    ("vacancy-defect-card.md", "engine_type"): (
        "随机引擎类型 (random engine type)，`0=Sobol`，`1=Uniform`。",
        "Uniform 在大批量生成时更快；Sobol 在样本较少时对空位数与位置的覆盖更均衡。样本很多时两者统计差异通常不大。",
        triplet("小样本覆盖优先 Sobol", "先 Uniform 提速，再用 Sobol 抽样复核分布", "超大样本阶段优先速度，固定引擎避免混杂偏差"),
    ),
    ("cell-strain-card.md", "engine_type"): (
        "应变轴模式 (strain axes mode)，可选 `uniaxial / biaxial / triaxial / isotropic`。",
        "决定同时施加应变的轴向组合与样本规模：轴数越多，组合空间越大、计算成本越高。",
        triplet("isotropic 或 uniaxial 基线", "biaxial 覆盖主要耦合响应", "triaxial 仅在预算充足且有明确需求时启用"),
    ),
    ("interstitial-adsorbate-card.md", "mode"): (
        "插入模式枚举 (insertion mode enum)。UI 下拉显示字符串选项，但配置序列化保存为整数索引：`0=Interstitial`，`1=Adsorption`。",
        "`Interstitial` 在晶胞内部随机采样候选位点；`Adsorption` 在选定表面法向上方按 `offset` 放置，并启用 `axis/offset` 参数。",
        triplet("先用 Interstitial 跑通最小距离与成功率", "按任务切换 Interstitial/Adsorption", "Adsorption 批量生成前固定 axis 并抽样检查表面位点"),
    ),
    ("interstitial-adsorbate-card.md", "insert_count"): (
        "每个生成结构插入的原子数 (atoms per generated structure)。",
        "该值越大，缺陷密度越高且碰撞失败概率上升；需与 `min_distance`、`max_attempts` 联合调节。",
        triplet("1-2", "2-6", "6-20"),
    ),
    ("interstitial-adsorbate-card.md", "structure_count"): (
        "每个输入结构生成的输出数量 (structures to generate per input)。",
        "直接决定该卡片输出规模与运行耗时。",
        triplet("10-100", "100-400", "400-1000"),
    ),
    ("interstitial-adsorbate-card.md", "min_distance"): (
        "候选插入位点与已有原子的最小距离阈值 (minimum allowed distance, `Å`)。",
        "阈值越大越保守、物理性更稳，但可行位点更少、成功率会下降。",
        triplet("1.6-2.5 Å", "1.2-1.6 Å", "0.8-1.2 Å（仅探索）"),
    ),
    ("interstitial-adsorbate-card.md", "max_attempts"): (
        "每个待插入原子的最大随机尝试次数 (maximum random attempts per atom)。",
        "提高该值可提升成功率，但会线性增加采样耗时。",
        triplet("50-200", "200-600", "600-1000"),
    ),
    ("interstitial-adsorbate-card.md", "seed"): (
        "随机种子值 (random seed value)。",
        "只影响随机插入路径，不改变物理判据。",
        triplet("0（随机）", "1-99（可复现）", "100-9999（多 seed 对比）"),
    ),
    ("interstitial-adsorbate-card.md", "axis"): (
        "表面法向轴枚举 (surface normal axis enum)。仅在 `mode=Adsorption` 时生效：`0=a(x)`，`1=b(y)`，`2=c(z)`。",
        "决定沿哪一条晶轴法向定义“表面上方”放置方向，会直接改变吸附位点分布。",
        triplet("先用 `c(z)` 与常见 slab 约定对齐", "按实际 slab 法向选择对应轴", "多轴探索前先可视化核查法向是否正确"),
    ),
    ("interstitial-adsorbate-card.md", "offset"): (
        "吸附放置高度偏移 (offset distance, `Å`)。",
        "仅在 `Adsorption` 模式生效；偏移越大越远离表面，局域相互作用通常减弱。",
        triplet("1.0-2.0 Å", "2.0-4.0 Å", "4.0-8.0 Å"),
    ),
    ("cell-strain-card.md", "x_range"): (
        "X 方向应变扫描区间（单位 `%`），格式为 `[min,max,step]`。",
        "按百分比应变扫描 x 轴（例如 `-5` 到 `5` 表示 `-5%` 到 `+5%`）。范围越宽或步长越小，组合数增长越快。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    ("cell-strain-card.md", "y_range"): (
        "Y 方向应变扫描区间（单位 `%`），格式为 `[min,max,step]`。",
        "按百分比应变扫描 y 轴（例如 `-5` 到 `5` 表示 `-5%` 到 `+5%`）。范围越宽或步长越小，组合数增长越快。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    ("cell-strain-card.md", "z_range"): (
        "Z 方向应变扫描区间（单位 `%`），格式为 `[min,max,step]`。",
        "按百分比应变扫描 z 轴（例如 `-5` 到 `5` 表示 `-5%` 到 `+5%`）。范围越宽或步长越小，组合数增长越快。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    ("shear-matrix-card.md", "xy_range"): (
        "XY 剪切分量扫描区间（单位 `%`），格式为 `[min,max,step]`。",
        "剪切矩阵分量按 `sxy/100` 写入，`sxy=5` 即 `0.05` 剪切分量。范围越宽或步长越小，生成组合越多。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    ("shear-matrix-card.md", "yz_range"): (
        "YZ 剪切分量扫描区间（单位 `%`），格式为 `[min,max,step]`。",
        "剪切矩阵分量按 `syz/100` 写入，`syz=5` 即 `0.05` 剪切分量。范围越宽或步长越小，生成组合越多。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    ("shear-matrix-card.md", "xz_range"): (
        "XZ 剪切分量扫描区间（单位 `%`），格式为 `[min,max,step]`。",
        "剪切矩阵分量按 `sxz/100` 写入，`sxz=5` 即 `0.05` 剪切分量。范围越宽或步长越小，生成组合越多。",
        triplet("±1-2%", "±3-5%", "±6%+"),
    ),
    ("shear-angle-card.md", "alpha_range"): (
        "Alpha 角扫描区间（单位 `°`），格式为 `[min,max,step]`。",
        "表示相对原始晶格角的增量 `Δalpha`；例如 `[-2,2,1]` 表示在 `-2°` 到 `+2°` 内按 `1°` 扫描。",
        triplet("±1°", "±3°", "±6°"),
    ),
    ("shear-angle-card.md", "beta_range"): (
        "Beta 角扫描区间（单位 `°`），格式为 `[min,max,step]`。",
        "表示相对原始晶格角的增量 `Δbeta`；例如 `[-2,2,1]` 表示在 `-2°` 到 `+2°` 内按 `1°` 扫描。",
        triplet("±1°", "±3°", "±6°"),
    ),
    ("shear-angle-card.md", "gamma_range"): (
        "Gamma 角扫描区间（单位 `°`），格式为 `[min,max,step]`。",
        "表示相对原始晶格角的增量 `Δgamma`；例如 `[-2,2,1]` 表示在 `-2°` 到 `+2°` 内按 `1°` 扫描。",
        triplet("±1°", "±3°", "±6°"),
    ),
    ("composition-sweep-card.md", "method"): (
        "成分点生成方法 (composition sampling method)，可选 `Grid` 或 `Sobol`。",
        "Grid 便于解释且步长可控；Sobol 属于低差异序列，在少样本/高阶(order>=4)时覆盖通常更稳。样本足够多时两者差异会减小。",
        triplet("少样本或高阶优先 Sobol", "Grid 先试跑，再用 Sobol 补覆盖盲区", "大样本阶段按可解释性与算力预算选择"),
    ),
    ("random-doping-card.md", "rules"): (
        "掺杂规则表 (doping rules)，字段含 `target/dopants/use/concentration/count/group`。",
        "决定替换对象、替换比例和局域范围，是化学分布主控参数。",
        triplet("单规则窄窗口", "2-3 条主规则", "多规则并行并做后筛"),
    ),
    ("random-vacancy-card.md", "rules"): (
        "空位规则表 (vacancy rules)，字段含 `element/count/group`。",
        "控制删原子元素与密度分布。",
        triplet("单元素低计数", "多元素中计数", "高计数需稳定性筛查"),
    ),
    ("conditional-replace-card.md", "condition"): (
        "空间条件表达式 (condition expression)，支持 x/y/z 与逻辑运算。",
        "命中区域越宽，替换越全局；越窄，越局域。",
        triplet("窄窗口验证", "分层范围", "all 全局替换仅探索"),
    ),
    ("conditional-replace-card.md", "replacements"): (
        "替换配方 (replacement map)，支持 `A:0.7,B:0.3` 或 JSON dict。",
        "决定替换后化学组成分布。",
        triplet("单元素", "双元素比例", "三元素以上需关注稀疏"),
    ),
    ("magmom-rotation-card.md", "max_angle"): (
        "最大旋转角 (max rotation angle)。",
        "主控磁方向扰动强度，角度越大偏离基态越远。",
        triplet("2-5°", "8-15°", "20°+ 需重点筛查"),
    ),
    ("magmom-rotation-card.md", "magnitude_factor"): (
        "模长缩放区间 (magnitude factor range)。",
        "区间越宽，磁矩长度分布越发散。",
        triplet("0.98-1.02", "0.95-1.05", "0.85-1.15"),
    ),
    ("vibration-perturb-card.md", "amplitude"): (
        "模态位移幅度 (mode displacement amplitude)。",
        "主控振动扰动强度，过大易进入高能异常区。",
        triplet("0.01-0.03", "0.04-0.07", "0.1+ 需后筛"),
    ),
    ("vibration-perturb-card.md", "modes_per_sample"): (
        "每样本叠加模态数 (modes per sample)。",
        "模态数越高，扰动方向组合越复杂。",
        triplet("1-2", "3-4", "5+"),
    ),
}


def split_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[name] = text[start:end].strip("\n")
    return out


def section_body(sections: dict[str, str], canonical: str) -> str:
    for alias in SECTION_ALIASES.get(canonical, [canonical]):
        if alias in sections:
            return sections[alias]
    return ""


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


RUNTIME_DEFAULT_CACHE: dict[str, dict[str, object]] = {}


def load_runtime_defaults(
    source_file: str,
    card_name: str,
    class_name: str,
) -> dict[str, object]:
    cache_key = f"{source_file}::{class_name}"
    if cache_key in RUNTIME_DEFAULT_CACHE:
        return RUNTIME_DEFAULT_CACHE[cache_key]
    _ensure_app()
    source_path = ROOT / source_file
    if not source_path.exists():
        RUNTIME_DEFAULT_CACHE[cache_key] = {}
        return {}
    spec = importlib.util.spec_from_file_location(
        f"_rewrite_card_{source_path.stem}_{abs(hash(source_file))}",
        source_path,
    )
    if spec is None or spec.loader is None:
        RUNTIME_DEFAULT_CACHE[cache_key] = {}
        return {}
    mod = importlib.util.module_from_spec(spec)
    original_register = CardManager.register_card
    CardManager.register_card = classmethod(lambda _cls, card_cls: card_cls)
    try:
        spec.loader.exec_module(mod)
    finally:
        CardManager.register_card = original_register
    cls = None
    for obj in mod.__dict__.values():
        if inspect.isclass(obj) and hasattr(obj, "card_name"):
            if getattr(obj, "card_name", None) == card_name:
                cls = obj
                break
    if cls is None and class_name and hasattr(mod, class_name):
        cls = getattr(mod, class_name)
    if cls is None:
        RUNTIME_DEFAULT_CACHE[cache_key] = {}
        return {}
    card = cls(None)
    defaults = dict(card.to_dict())
    RUNTIME_DEFAULT_CACHE[cache_key] = defaults
    return defaults


def normalize_default_for_docs(key: str, value: object) -> object:
    if key != "nep_path" or not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return ""
    # Prefer project-relative path in docs to avoid machine-specific absolute paths.
    try:
        rel = Path(raw).resolve().relative_to(ROOT.resolve())
        return rel.as_posix()
    except Exception:
        pass
    normalized = raw.replace("\\", "/")
    if "src/NepTrainKit/Config/nep89.txt" in normalized:
        return "src/NepTrainKit/Config/nep89.txt"
    if normalized.endswith("Config/nep89.txt"):
        return "src/NepTrainKit/Config/nep89.txt"
    return normalized


def format_default_cell(value: object, key: str | None = None) -> str:
    if key is not None:
        value = normalize_default_for_docs(key, value)
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{'true' if value else 'false'}`"
    if isinstance(value, str):
        return f"`{json.dumps(value, ensure_ascii=False)}`"
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (int, float, list, dict)):
        return f"`{json.dumps(value, ensure_ascii=False)}`"
    return f"`{str(value)}`"


def parse_control_rows(control: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []

    # Legacy table format.
    lines = [ln for ln in control.splitlines() if ln.strip().startswith("|")]
    if lines:
        for ln in lines[2:]:
            cols = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cols) < 7:
                continue
            ui, key, typ, default = cols[:4]
            rows.append((ui, key, typ, default))
        if rows:
            return rows

    # New list format.
    blocks = list(
        re.finditer(
            r"^###\s+`(?P<key>[^`]+)`\s+\((?P<ui>[^)]+)\)\s*$",
            control,
            flags=re.MULTILINE,
        )
    )
    for i, m in enumerate(blocks):
        key = m.group("key").strip()
        ui = m.group("ui").strip()
        start = m.end()
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(control)
        chunk = control[start:end]
        typ_match = re.search(r"^- 类型/范围 \(Type/Range\):\s*(.+)$", chunk, flags=re.MULTILINE)
        default_match = re.search(r"^- 默认值 \(Default\):\s*(.+)$", chunk, flags=re.MULTILINE)
        typ = typ_match.group(1).strip() if typ_match else "unknown"
        default = default_match.group(1).strip() if default_match else "`null`"
        rows.append((ui, f"`{key}`", typ, default))
    return rows


def key_info(card_file: str, key_raw: str, ui: str) -> tuple[str, str, str]:
    key = key_raw.strip("` ")
    if (card_file, key) in CARD_KEY_OVERRIDE:
        return CARD_KEY_OVERRIDE[(card_file, key)]
    if key in GENERIC_KEY_INFO:
        return GENERIC_KEY_INFO[key]
    if key.endswith("_range"):
        return (
            f"`{key}` 为扫描范围参数 (range parameter)。",
            "范围决定采样覆盖宽度，步长决定分辨率与样本数量。",
            triplet("窄范围+粗步长", "中范围+中步长", "宽范围+细步长需预算"),
        )
    if key.endswith("_condition"):
        return (
            f"`{key}` 为约束条件参数 (condition parameter)。",
            "用于限定采样边界或输出规模。",
            triplet("保守边界", "平衡边界", "激进边界需后筛"),
        )
    if key.endswith("_radio_button"):
        return (
            f"`{key}` 为模式开关 (mode toggle)。",
            "决定当前执行路径是否启用。",
            triplet("先启用单一路径", "按任务切换", "多路径并开需验证优先级"),
        )
    if key.startswith("afm_"):
        return (
            f"`{key}` 为 AFM 分支参数 (AFM branch parameter)。",
            "影响反铁磁子晶格划分与方向分布。",
            triplet("标准配置", "按结构微调", "复杂配置需可视化核查"),
        )
    if key.startswith("pm_"):
        return (
            f"`{key}` 为 PM 分支参数 (PM branch parameter)。",
            "影响顺磁方向采样与样本规模。",
            triplet("保守 PM", "平衡 PM", "激进 PM 需过滤"),
        )
    if key.startswith("super_"):
        return (
            f"`{key}` 为超胞模式参数 (supercell mode parameter)。",
            "影响扩胞策略与规模控制。",
            triplet("先单模式", "双模式对比", "多模式并行需预算"),
        )
    return (
        f"参数 `{key}` 控制 `{ui}` 的执行语义 (operation semantics)。",
        "改变该参数会影响样本分布、物理风险或计算成本。",
        triplet("默认值先验证", "围绕默认值微调", "大偏移调参需配质量筛选"),
    )


def build_control(card_file: str, old_control: str, runtime_defaults: dict[str, object]) -> str:
    rows = parse_control_rows(old_control)
    out: list[str] = []
    for ui, key_raw, typ, default in rows:
        key = key_raw.strip("` ")
        ui_display = CARD_UI_LABEL_OVERRIDE.get((card_file, key), ui)
        caption = CARD_UI_CAPTION_OVERRIDE.get((card_file, key), ui_display)
        set_range_hint = CARD_SET_RANGE_HINT.get((card_file, key), "")
        typ_display = display_type_range(card_file, key, typ, runtime_defaults)
        default_value = runtime_defaults.get(key)
        default_cell = format_default_cell(runtime_defaults[key], key=key) if key in runtime_defaults else default
        m, e, r = key_info(card_file, key_raw, ui_display)
        style = recommendation_style(card_file, key, typ_display)
        if style == "tiered" and is_numeric_tiered_type(typ_display) and not recommendation_tiers_have_numbers(r):
            r = auto_numeric_triplet(key, default_value)
        out.extend(
            [
                f"### `{key}` ({ui_display})",
                f"- UI Label: `{ui_display}`",
                f"- 字段映射 (Field mapping): 序列化键 `{key}` <-> 界面标签 `{ui_display}`。",
                f"- 控件标签 (Caption): `{caption}`。",
                f"- 控件解释 (Widget): {widget_hint(typ_display)}",
                f"- 类型/范围 (Type/Range): {typ_display}",
                f"- 默认值 (Default): {default_cell}",
                f"- 含义 (Meaning): {m}",
                f"- 对输出规模/物理性的影响: {e}",
            ]
        )
        if set_range_hint:
            out.append(f"- 控件范围 (setRange): `{set_range_hint}`。")
        if style == "tiered":
            out.extend(
                [
                    "- 推荐范围 (Recommended range):",
                    *[f"  - {line}" for line in r.splitlines()],
                    "",
                ]
            )
        elif style == "binary":
            on_note, off_note = bool_note(key, ui)
            out.extend(
                [
                    "- 配置建议 (Practical note):",
                    f"  - 开启：{on_note}",
                    f"  - 关闭：{off_note}",
                    "",
                ]
            )
        else:
            out.extend(
                [
                    f"- 配置建议 (Practical note): {single_note(key, ui, typ_display)}",
                    "",
                ]
            )

    if card_file == "random-doping-card.md":
        out += [
            "### 规则输入 Schema (Rule input schema)",
            "`rules` 在配置中保存为 JSON 字符串，语义为 rule object 列表。",
            "- `target` (string): 被替换元素。",
            "- `dopants` (object): 掺杂元素及权重，例如 `{\"Ge\":0.7,\"C\":0.3}`。",
            "- `use` (string): `concentration` 或 `count`。",
            "- `concentration` (list[2]): 浓度区间。",
            "- `count` (list[2]): 替换数量区间。",
            "- `group` (list[string], optional): 仅作用于指定 group。",
        ]
    if card_file == "random-vacancy-card.md":
        out += [
            "### 规则输入 Schema (Rule input schema)",
            "`rules` 在配置中保存为 JSON 字符串，语义为 vacancy rule 列表。",
            "- `element` (string): 删除目标元素。",
            "- `count` (list[2]): 删除数量区间。",
            "- `group` (list[string], optional): 仅作用于指定 group。",
        ]
    if card_file == "conditional-replace-card.md":
        out += [
            "### 替换输入 Schema (Replacement input schema)",
            "- `replacements` 支持 `Co:0.7,Ni:0.3` 或 JSON dict 字符串。",
            "- `condition` 支持 `x/y/z` 与 `and/or/not` 逻辑表达式。",
            "- 建议先用 `all` 验证替换路径，再收紧局域条件。",
        ]
    return "\n".join(out)


def build_when(card_file: str, profile: dict[str, list[str] | str]) -> str:
    lines = [f"- {s}" for s in profile["when"]]  # type: ignore[index]
    if card_file != "card-group.md":
        lines.append("> 物理提示 (Physics caution): 上调幅度前先抽查最短键长、异常角度和晶胞条件数。")
    return "\n".join(lines)


def build_list(profile: dict[str, list[str] | str], key: str) -> str:
    return "\n".join(f"- {s}" for s in profile[key])  # type: ignore[index]


def build_troubleshooting(card_file: str, profile: dict[str, list[str] | str]) -> str:
    lines = [f"- {x}" for x in profile["fail"]]  # type: ignore[index]
    if card_file != "card-group.md":
        lines.extend(
            [
                "- 参数冲突导致行为异常：先保留一条主控制路径（一个 mode + 一组主幅度参数）。",
                "- 输出分布不符合目标：抽样检查后再回调关键参数。",
            ]
        )
    return "\n".join(lines)


CARD_FORMULAS: dict[str, str] = {
    "cell-strain-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\epsilon_i=\\frac{s_i}{100},\\quad \\mathbf{C}'=\\mathbf{D}\\mathbf{C},\\quad \\mathbf{D}=\\mathrm{diag}(1+\\epsilon_x,1+\\epsilon_y,1+\\epsilon_z)$$",
            "$$\\text{isotropic: }\\mathbf{C}'=(1+\\epsilon)\\mathbf{C}$$",
        ]
    ),
    "cell-scaling-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$f_k\\in[1-m,1+m],\\quad a_i'=f_i a_i$$",
            "$$\\theta_j'=g_j\\theta_j\\quad(\\text{when }perturb\\_angle=true)$$",
            "$$\\mathbf{b}'=[b'\\cos\\gamma',\\ b'\\sin\\gamma',\\ 0]$$",
            "$$c_x'=c'\\cos\\beta',\\quad c_y'=\\frac{c'(\\cos\\alpha'-\\cos\\beta'\\cos\\gamma')}{\\sin\\gamma'},\\quad c_z'=\\sqrt{c'^2-c_x'^2-c_y'^2}$$",
            "$$\\mathbf{c}'=[c_x',\\ c_y',\\ c_z']$$",
        ]
    ),
    "shear-angle-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\alpha'=\\alpha+\\Delta\\alpha,\\quad \\beta'=\\beta+\\Delta\\beta,\\quad \\gamma'=\\gamma+\\Delta\\gamma$$",
            "$$\\mathbf{C}'=\\mathrm{cellpar\\_to\\_cell}(a,b,c,\\alpha',\\beta',\\gamma')$$",
        ]
    ),
    "shear-matrix-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\gamma_{xy}=\\frac{s_{xy}}{100},\\ \\gamma_{yz}=\\frac{s_{yz}}{100},\\ \\gamma_{xz}=\\frac{s_{xz}}{100}$$",
            "$$\\mathbf{S}=\\begin{bmatrix}1&\\gamma_{xy}&\\gamma_{xz}\\\\0&1&\\gamma_{yz}\\\\0&0&1\\end{bmatrix},\\quad \\mathbf{C}'=\\mathbf{C}\\mathbf{S}$$",
            "$$\\text{symmetric=true 时再加 }S_{21}=\\gamma_{xy},\\ S_{32}=\\gamma_{yz},\\ S_{31}=\\gamma_{xz}$$",
        ]
    ),
    "super-cell-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\mathbf{T}=\\mathrm{diag}(n_a,n_b,n_c),\\quad \\mathbf{C}'=\\mathbf{C}\\mathbf{T}$$",
            "$$N'=N\\cdot n_a n_b n_c$$",
            "$$n_a^{(\\max)}=\\max\\left(\\left\\lfloor\\frac{L_a^*}{\\lVert\\mathbf{a}\\rVert}\\right\\rfloor,1\\right),\\quad n_a^{(\\min)}=\\left\\lfloor\\frac{L_a^*}{\\lVert\\mathbf{a}\\rVert}\\right\\rfloor+1$$",
        ]
    ),
    "perturb-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\Delta\\mathbf{r}_i=\\boldsymbol\\xi_i\\odot d_i,\\quad \\boldsymbol\\xi_i\\in[-1,1]^3,\\quad \\mathbf{r}_i'=\\mathbf{r}_i+\\Delta\\mathbf{r}_i$$",
            "$$\\text{organic cluster 模式: }\\forall j\\in\\mathcal{C},\\ \\mathbf{r}_j'=\\mathbf{r}_j+\\Delta\\mathbf{r}_{\\text{anchor}(\\mathcal{C})}$$",
        ]
    ),
    "vibration-perturb-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\mathbf{r}'=\\mathbf{r}+A\\sum_{k\\in\\mathcal{K}} c_k\\mathbf{u}_k$$",
            "$$c_k\\sim\\mathcal{N}(0,1)\\ \\text{or}\\ \\mathcal{U}(-1,1),\\quad c_k\\leftarrow\\frac{c_k}{\\sqrt{|\\omega_k|}}\\ (\\text{when scale\\_by\\_frequency=true})$$",
        ]
    ),
    "magmom-rotation-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\mathbf{m}'=\\lambda\\,\\mathbf{R}(\\hat{\\mathbf{n}},\\theta)\\,\\mathbf{m},\\quad \\lambda\\in[f_{\\min},f_{\\max}]$$",
            "$$\\mathbf{R}(\\hat{\\mathbf{n}},\\theta)=\\cos\\theta\\,\\mathbf{I}+(1-\\cos\\theta)\\hat{\\mathbf{n}}\\hat{\\mathbf{n}}^\\top+\\sin\\theta\\,[\\hat{\\mathbf{n}}]_{\\times}$$",
        ]
    ),
    "group-label-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$g_{k\\text{-vec}}=\\left\\lfloor 2(\\mathbf{s}\\cdot\\mathbf{k})\\right\\rfloor\\bmod 2$$",
            "$$g_{parity}=\\left(\\mathrm{round}(2s_x)+\\mathrm{round}(2s_y)+\\mathrm{round}(2s_z)\\right)\\bmod 2$$",
        ]
    ),
    "fps-filter-card.md": "\n".join(
        [
            "### 关键公式 (Core equations)",
            "$$\\mathbf{d}_i=\\mathrm{NEP89}(x_i)$$",
            "$$i_t=\\arg\\max_j\\ \\min_{i\\in S_{t-1}}\\lVert\\mathbf{d}_j-\\mathbf{d}_i\\rVert_2,\\quad S_t=S_{t-1}\\cup\\{i_t\\}$$",
            "$$\\min_{i\\in S_t,j\\in S_t,i\\ne j}\\lVert\\mathbf{d}_i-\\mathbf{d}_j\\rVert_2\\ge d_{\\min}\\ (\\text{if feasible})$$",
        ]
    ),
}

CARD_IMPORTANT_NOTES: dict[str, str] = {
    "composition-sweep-card.md": "\n".join(
        [
            ":::{important}",
            "该卡片只在 `info` 中添加组分信息，并不实际生成合金结构，需要配合 `Random Occupancy` 使用。",
            ":::",
        ]
    ),
}


GENERIC_MINIMAL_EXAMPLE = "最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构，先检查生成的 tags/arrays，再放大规模。"


def build_function_desc(card_file: str, profile: dict[str, list[str] | str]) -> str:
    base = str(profile["does"]).strip()
    important = CARD_IMPORTANT_NOTES.get(card_file, "").strip()
    formula = CARD_FORMULAS.get(card_file, "").strip()
    chunks = [base]
    if important:
        chunks.append(important)
    chunks.append(build_quickstart_block(card_file))
    if formula:
        chunks.append(formula)
    return "\n\n".join(chunks)


FPS_RECOMMENDED_FLOW = (
    "建议先导出 xyz，导入第一个模块使用 `nep89` 预测并剔除非物理结构，"
    "再执行最远点采样（FPS）。"
)
FPS_RECOMMENDED_TIP = "\n".join(
    [
        ":::{tip}",
        f"高通量示例：{FPS_RECOMMENDED_FLOW}",
        ":::",
    ]
)


CARD_GROUP_MINIMAL_EXAMPLE = (
    "最小可运行示例：在 Card Group 内放入两张互不依赖的分支卡片，"
    "共享同一输入运行后核对汇总输出条数。"
)
CARD_GROUP_THROUGHPUT_EXAMPLE = (
    "高通量示例：先在组内组织共享输入的多分支生成，"
    "再在组外串接清洗/采样链路（例如 NEP89 清洗与 FPS），避免把依赖链写进组内。"
)

GROUP_LABEL_MINIMAL_EXAMPLE = "最小可运行示例：对单帧执行后检查 `atoms.arrays['group']` 是否写入且标签与预期一致。"
GROUP_LABEL_THROUGHPUT_EXAMPLE = (
    "高通量示例：先统一 `group_a/group_b` 命名规范，再批量生成并抽样核对下游卡片读取是否一致。"
)


def build_quickstart_block(card_file: str) -> str:
    if card_file == "card-group.md":
        minimal = CARD_GROUP_MINIMAL_EXAMPLE
        throughput = CARD_GROUP_THROUGHPUT_EXAMPLE
    elif card_file == "group-label-card.md":
        minimal = GROUP_LABEL_MINIMAL_EXAMPLE
        throughput = GROUP_LABEL_THROUGHPUT_EXAMPLE
    else:
        minimal = GENERIC_MINIMAL_EXAMPLE
        throughput = f"高通量示例：{FPS_RECOMMENDED_FLOW}"
    return "\n".join(
        [
            "### 快速上手",
            minimal,
            "",
            ":::{tip}",
            throughput,
            ":::",
        ]
    )


CARD_GROUP_PRESETS_SECTION = "\n".join(
    [
        "### 保守（Safe）",
        "```json",
        "{",
        '  "class": "CardGroup",',
        '  "check_state": true,',
        '  "card_list": [],',
        '  "filter_card": {}',
        "}",
        "```",
        "",
        "### 平衡（Balanced）",
        "```json",
        "{",
        '  "class": "CardGroup",',
        '  "check_state": true,',
        '  "card_list": [],',
        '  "filter_card": {}',
        "}",
        "```",
        "",
        "### 激进/探索（Aggressive/Exploration）",
        "```json",
        "{",
        '  "class": "CardGroup",',
        '  "check_state": true,',
        '  "card_list": [],',
        '  "filter_card": {}',
        "}",
        "```",
    ]
)

GROUP_LABEL_PRESETS_SECTION = "\n".join(
    [
        "### 基础模板（Baseline）",
        "```json",
        "{",
        '  "class": "GroupLabelCard",',
        '  "check_state": true,',
        '  "mode": "k-vector layers (recommended)",',
        '  "kvec": "111",',
        '  "group_a": "A",',
        '  "group_b": "B",',
        '  "overwrite": true',
        "}",
        "```",
        "",
        "### 兼容模板（Compatible）",
        "```json",
        "{",
        '  "class": "GroupLabelCard",',
        '  "check_state": true,',
        '  "mode": "fractional parity (2x rounding)",',
        '  "kvec": "111",',
        '  "group_a": "A",',
        '  "group_b": "B",',
        '  "overwrite": false',
        "}",
        "```",
        "",
        "### 自定义模板（Custom）",
        "```json",
        "{",
        '  "class": "GroupLabelCard",',
        '  "check_state": true,',
        '  "mode": "k-vector layers (recommended)",',
        '  "kvec": "110",',
        '  "group_a": "S1",',
        '  "group_b": "S2",',
        '  "overwrite": true',
        "}",
        "```",
    ]
)


def normalize_presets_section(text: str, card_file: str) -> str:
    out = text
    out = re.sub(
        r"^###\s*Safe\s*$",
        "### 保守（Safe）",
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r"^###\s*Balanced\s*$",
        "### 平衡（Balanced）",
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r"^###\s*Aggressive/Exploration\s*$",
        "### 激进/探索（Aggressive/Exploration）",
        out,
        flags=re.MULTILINE,
    )
    if card_file == "card-group.md":
        return CARD_GROUP_PRESETS_SECTION
    if card_file == "group-label-card.md":
        return GROUP_LABEL_PRESETS_SECTION

    # Keep presets section focused on copy-paste JSON only.
    out = re.sub(
        r"^最小可运行示例.*$",
        "",
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r"^高通量示例.*$",
        "",
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r":::\{tip\}\s*\n\s*:::",
        "",
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


COMBO_DESC_CN: dict[str, str] = {
    "AFM group mode requires stable group labels.": "AFM 分组模式需要稳定的 group 标签。",
    "add local noise around faulted geometries.": "在层错构型周围补充局部扰动。",
    "add local variance to each strained frame.": "对每个应变帧增加局部差异。",
    "add mild cell changes after mode displacement.": "在模态位移后追加轻量晶胞变化。",
    "apply rule constraints to selected regions.": "在选定区域施加规则约束。",
    "avoid over-representing similar angle states.": "避免相近角度状态被过度采样。",
    "broad strain grid with bounded export size.": "在保证导出规模可控的前提下扩展应变网格。",
    "build complementary defect families.": "构建互补缺陷族样本。",
    "build warped multilayers then sample interstitial/adsorbate sites.": "先构建层间形变结构，再采样插隙/吸附位点。",
    "canonical HEA workflow.": "这是典型的高熵合金工作流。",
    "co-sample vacancy and interstitial families.": "联合采样空位与插隙缺陷族。",
    "combine broad composition space with localized substitutions.": "将大范围成分空间与局域替位结合。",
    "combine global deformation with local displacement noise.": "结合全局形变与局部位移噪声。",
    "combine torsional diversity with mild cell variability.": "将扭转多样性与轻量晶胞变化结合。",
    "complete magnetic workflow for AFM/FM/PM datasets.": "形成覆盖 AFM/FM/PM 的完整磁性流程。",
    "composition-first then site-specific substitution.": "先做成分展开，再做位点定向替位。",
    "control combinatorial growth from range grids.": "控制范围网格引起的组合爆炸。",
    "controlled vacancy fractions at larger system sizes.": "在较大体系中可控地采样空位比例。",
    "convert target composition tags into explicit atom assignments.": "将目标成分标签转换为显式原子占位。",
    "couple cell perturbation with atomic perturbation.": "将晶格扰动与原子扰动联用。",
    "cover prototype + strain phase space efficiently.": "高效覆盖原型结构与应变相空间。",
    "create surface vacancy datasets with explicit element/group control.": "在显式元素/group 控制下构建表面空位数据集。",
    "curate diverse conformers after torsion sampling.": "在扭转采样后筛选多样构象。",
    "deduplicate structures after coordinate-conditioned edits.": "对坐标条件编辑后的结构执行去重。",
    "down-select magnetic variants after structural generation.": "在结构生成后下采样磁性变体。",
    "filter merged branch outputs before top-level export.": "在顶层导出前先过滤分支汇合结果。",
    "generate clean templates, then decorate compositionally.": "先生成干净模板，再进行成分修饰。",
    "generate complementary defect families from same input.": "基于同一输入生成互补缺陷族。",
    "generate ordered seeds first, then sample orientation variations.": "先生成有序种子，再采样方向变化。",
    "generate realistic defective surfaces.": "生成更接近实际的缺陷表面样本。",
    "geometry-gated replacement followed by wider substitution sampling.": "先做几何门控替换，再做更广泛替位采样。",
    "grow the system first, then scan strain states and down-select.": "先扩胞，再扫描应变状态并下采样。",
    "keep broad yet non-redundant perturbation coverage.": "保持覆盖广度同时避免冗余扰动。",
    "keep enough atoms after defect deletion.": "保证删缺陷后仍有足够原子数。",
    "keep representative slab orientations and thicknesses.": "保留有代表性的 slab 取向与厚度。",
    "parallel diversity then global down-selection.": "先并行扩展多样性，再全局下采样。",
    "place FPS at branch end to cap export redundancy.": "将 FPS 放在分支末端以限制导出冗余。",
    "preserve enough atoms while exploring vacancy rules.": "在探索空位规则时保留足够原子数。",
    "preserve sublattice context before moment perturbation.": "在磁矩扰动前保留子晶格上下文。",
    "reduce near-duplicate slip steps.": "减少近重复的滑移步长样本。",
    "refine sheared structures with local displacements.": "用局部位移进一步细化剪切结构。",
    "remove redundant mode-sampled structures.": "去除模态采样产生的冗余结构。",
    "restrict occupancy changes to chosen groups only.": "将占位变化限制在指定 group。",
    "robust baseline for thermal-like local disorder.": "作为热扰动类局部无序的稳健基线。",
    "scan angle first, then perturb lengths lightly.": "先扫描角度，再轻量扰动晶格长度。",
    "standard adsorbate/interstitial surface workflow.": "标准的吸附/插隙表面流程。",
    "target specific sublattices/layers by group.": "按 group 定向作用于特定子晶格/层。",
    "trim large stacked outputs before export.": "导出前裁剪过大的层叠输出。",
}


def normalize_combos_section(text: str, card_file: str) -> str:
    if card_file == "card-group.md":
        return "\n".join(
            [
                "- Card Group(Atomic Perturb, Lattice Perturb, Shear Matrix Strain) -> 组外过滤链路: 组内先生成共享输入的多分支结果，再在组外统一清洗与采样。",
                "- Card Group(Random Vacancy, Insert Defect) -> export: 适合从同一输入并行生成互补缺陷族；若需要严格依赖链，请拆回顶层顺序卡片。",
            ]
        )
    if card_file == "fps-filter-card.md":
        return "\n".join(
            [
                "- 任意生成分支 -> 本卡: 建议把下采样放在分支末端，统一控制导出冗余。",
                "- Card Group -> 本卡（组外）: 先汇总分支结果，再执行距离约束采样，避免组内依赖混乱。",
            ]
        )
    if card_file == "cell-strain-card.md":
        return "\n".join(
            [
                "- Lattice Strain -> Atomic Perturb: 对每个应变帧增加局部差异。",
                "- Lattice Strain -> Shear Matrix Strain: 在轴向应变后补充非对角剪切分量覆盖。",
            ]
        )
    if card_file == "composition-sweep-card.md":
        return "\n".join(
            [
                "- Composition Sweep -> Random Occupancy: 将目标成分标签转换为显式原子占位。",
                "- Composition Sweep -> Random Doping: 先做成分展开，再做位点定向替位。",
            ]
        )
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and ": " in stripped:
            body = stripped[2:]
            prefix, desc = body.split(": ", 1)
            if "FPS Filter" in prefix:
                continue
            desc_cn = COMBO_DESC_CN.get(desc.strip(), desc.strip())
            out_lines.append(f"- {prefix}: {desc_cn}")
        else:
            out_lines.append(line)
    bullet_count = sum(1 for ln in out_lines if ln.strip().startswith("- "))
    fillers = [
        "- 当前卡片 -> 目标变换卡: 先完成本卡输出，再叠加下一类结构变换扩展覆盖。",
        "- 当前卡片 -> 导出: 导出前抽样检查标签与结构质量，避免异常样本进入训练集。",
    ]
    i = 0
    while bullet_count < 2 and i < len(fillers):
        out_lines.append(fillers[i])
        bullet_count += 1
        i += 1
    return "\n".join(out_lines)


def normalize_output_tags_section(text: str) -> str:
    out = re.sub(
        r"Config_type tag patterns emitted by this card:",
        "该卡片输出的 Config_type 标签模式：",
        text,
        flags=re.IGNORECASE,
    )
    replacements = {
        "No dedicated Config_type tag is emitted by this card.": "该卡片本身不新增专用 Config_type 标签。",
        "Stores nested card definitions under `card_list` and optional `filter_card` in card JSON.": "在卡片 JSON 中保存嵌套 `card_list` 定义及可选 `filter_card`。",
        "Creates/overwrites `atoms.arrays['group']` labels.": "创建/覆盖 `atoms.arrays['group']` 标签数组。",
    }
    for en, zh in replacements.items():
        out = out.replace(en, zh)
    return out


def build_repro(serialized_keys: list[str]) -> str:
    if "use_seed" in serialized_keys and "seed" in serialized_keys:
        return "\n".join(
            [
                "- 设置 `use_seed=true` 且固定 `seed`，可在相同输入顺序下复现实验。",
                "- 上游随机卡片或输入顺序变化仍会改变最终样本集合。",
                "- 建议把 seed 与 pipeline 配置一起版本化记录。",
            ]
        )
    if "seed" in serialized_keys:
        return "\n".join(
            [
                "- `seed=[0]` 一般表示随机路径；固定非零值用于可重复对比。",
                "- 输入顺序变化会改变样本对应关系。",
            ]
        )
    return "\n".join(
        [
            "- 该卡片本身无显式随机种子，参数与输入一致时结果应确定。",
            "- 若上游含随机操作，仍需在 pipeline 层统一控制随机性。",
        ]
    )


def rewrite_one(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    first_h2 = re.search(r"^##\s+", text, flags=re.MULTILINE)
    if not first_h2:
        return
    preamble = text[: first_h2.start()].rstrip("\n")
    if path.name in CARD_TITLE_CN:
        preamble = re.sub(r"^#\s+.+$", f"# {CARD_TITLE_CN[path.name]}", preamble, count=1, flags=re.MULTILINE)

    sections = split_sections(text)
    class_match = re.search(r"`Class`:\s*`([^`]+)`", text)
    class_name = class_match.group(1) if class_match else ""
    profile = ensure_profile(path.name)

    meta_match = re.search(r"card-schema:\s*(\{.*\})\s*-->", text)
    meta = json.loads(meta_match.group(1)) if meta_match else {}
    serialized_keys = list(meta.get("serialized_keys", []))
    runtime_defaults = (
        load_runtime_defaults(
            str(meta.get("source_file", "")),
            str(meta.get("card_name", "")),
            class_name,
        )
        if class_name and meta.get("source_file")
        else {}
    )

    rebuilt_sections: dict[str, str] = {
        "功能说明": build_function_desc(path.name, profile),
        "适用场景与不适用场景": build_when(path.name, profile),
        "输入前提": build_list(profile, "prereq"),
        "参数说明（完整）": build_control(path.name, section_body(sections, "参数说明（完整）"), runtime_defaults),
        "推荐预设（可直接复制 JSON）": normalize_presets_section(
            section_body(sections, "推荐预设（可直接复制 JSON）"), path.name
        ),
        "推荐组合": normalize_combos_section(section_body(sections, "推荐组合"), path.name),
        "常见问题与排查": build_troubleshooting(path.name, profile),
        "输出标签 / 元数据变更": normalize_output_tags_section(section_body(sections, "输出标签 / 元数据变更")),
        "可复现性说明": build_repro(serialized_keys),
    }

    rebuilt = [preamble]
    for h in SECTION_ORDER:
        body = rebuilt_sections.get(h, "").strip("\n")
        rebuilt.append(f"\n## {h}\n{body}\n")

    path.write_text("\n".join(rebuilt).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    files = sorted(CARD_DOC_DIR.glob("*.md"))
    for p in files:
        rewrite_one(p)
    print(f"rewritten {len(files)} card pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
