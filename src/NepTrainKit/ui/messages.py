#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Global UI message center (Qt-based)

from PySide6.QtCore import QObject, Signal, Qt, QCoreApplication
from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition, MessageBox
from loguru import logger

from NepTrainKit.core.cards.errors import CardOperationError


def _tr(text: str) -> str:
    return QCoreApplication.translate("MessageManager", text)


def _zh_runtime_messages_enabled() -> bool:
    return QCoreApplication.translate("RuntimeMessage", "__language_probe__") == "zh_CN"


# Compatibility-only fallback for legacy raw strings. New card errors must use
# CardOperationError so wording changes do not require another substring rule.
_RUNTIME_PREFIX_REPLACEMENTS = (
    ("Failed to load dataset: ", "加载数据集失败："),
    ("NEP calculation failed [", "NEP 计算失败 ["),
    ("NEP Auto selected CPU because CUDA is unavailable ", "NEP 自动选择 CPU，因为 CUDA 不可用 "),
    ("Error occurred: ", "发生错误："),
    ("Failed to load ", "加载失败："),
    ("Unknown card class: ", "未知卡片类："),
    ("Imported preset: ", "已导入预设："),
    ("Preset exported to ", "预设已导出到 "),
    ("Deleted preset: ", "已删除预设："),
    ("Baseline preset saved: ", "基准预设已保存："),
    ("Preset did not match current dataset ", "预设与当前数据集不匹配 "),
    ("Preset shifted ", "预设已平移 "),
    ("Image exported to: ", "图片已导出到："),
    ("Data exported to: ", "数据已导出到："),
    ("Exported dataset summary to: ", "数据集摘要已导出到："),
    ("File exported to: ", "文件已导出到："),
    ("NEPKit currently does not support model_type=", "NEPKit 暂不支持 model_type="),
    ("Search completer candidates exceed ", "搜索候选项超过 "),
    ("Unsupported search type: ", "不支持的搜索类型："),
    ("Unknown element symbol: ", "未知元素符号："),
    ("An error occurred while running NEP calculator: ", "运行 NEP 计算器时出错："),
    ("RandomOccupancy: invalid composition: ", "RandomOccupancy：成分无效："),
    ("SpinSpiral: invalid magmom map: ", "SpinSpiral：magmom 映射无效："),
    ("FoldedHelix: invalid magmom map: ", "FoldedHelix：magmom 映射无效："),
    ("SetMagneticMoments: invalid magmom map: ", "SetMagneticMoments：magmom 映射无效："),
    ("SmallAngleSpinTilt: invalid magmom map: ", "SmallAngleSpinTilt：magmom 映射无效："),
    ("MagneticOrder: invalid magmom map: ", "MagneticOrder：magmom 映射无效："),
    ("Solvent Box Fill:", "周期溶剂盒："),
    ("LayerCopy:", "分层堆叠："),
    ("FPS Filter:", "FPS 代表性采样："),
    ("Geometry Filter:", "几何健全性过滤："),
    ("CompositionSweep Grid: ", "CompositionSweep 网格模式："),
    ("dz test failed: ", "dz 测试失败："),
    ("dz test ok: ", "dz 测试通过："),
)


_RUNTIME_TEXT_REPLACEMENTS = (
    ("NEP Auto selected CUDA acceleration for this model.", "NEP 自动为当前模型启用了 CUDA 加速。"),
    ("Dipole and polarizability models are CPU-only; NepTrainKit will use CPU regardless of the selected NEP backend.", "偶极矩和极化率模型仅支持 CPU；无论 NEP 后端设置为何，NepTrainKit 都会使用 CPU。"),
    ("DFT-D3 calculations are CPU-only; NepTrainKit will use CPU regardless of the selected NEP backend.", "DFT-D3 计算仅支持 CPU；无论 NEP 后端设置为何，NepTrainKit 都会使用 CPU。"),
    ("Loading existing official NEP .out files without opening the model.", "已直接加载现有的官方 NEP .out 文件，无需打开模型。"),
    (" descriptor.out is missing, so descriptor plots and FPS are unavailable. Install a nep-adapters version that supports this model to generate descriptors.", " 缺少 descriptor.out，因此描述符图和 FPS 暂不可用。如需生成描述符，请安装支持该模型的 nep-adapters 版本。"),
    ("If official NEP .out files already exist, keep a complete set of energy, force, virial, and stress or mforce outputs in the dataset directory.", "如果已有官方 NEP .out 文件，请在数据集目录中保留完整的 energy、force、virial，以及 stress 或 mforce 输出。"),
    ("The calculation will continue on CPU. ", "本次计算将继续使用 CPU。"),
    ("To enable CUDA, install a Linux CPU+CUDA nep-adapters wheel with a compatible NVIDIA driver.", "如需启用 CUDA，请在 Linux 上安装包含 CPU 和 CUDA 后端的 nep-adapters wheel，并使用兼容的 NVIDIA 驱动。"),
    ("Current canvas backend is vispy, but vispy canvas failed to initialize; fallback to pyqtgraph.", "当前画布后端是 vispy，但 vispy 画布初始化失败；已回退到 pyqtgraph。"),
    ("No data selected!", "未选择数据！"),
    ("NEP data has not been loaded yet!", "尚未加载 NEP 数据！"),
    ("No active structures to summarise.", "没有可汇总的有效结构。"),
    ("No active structures to scan.", "没有可扫描的有效结构。"),
    ("No active structures to export.", "没有可导出的有效结构。"),
    ("No removed structures to export.", "没有可导出的已移除结构。"),
    ("No bad structures tagged.", "没有标记为异常的结构。"),
    ("Please select some structures first!", "请先选择一些结构！"),
    ("Please enter a search query.", "请输入搜索条件。"),
    ("unsupported file format", "不支持的文件格式"),
    ("No vector data available", "没有可用的矢量数据"),
    ("The index is invalid, perhaps the structure has been deleted", "索引无效，结构可能已被删除。"),
    ("Failed to switch NEP model", "切换 NEP 模型失败"),
    ("Failed to build dataset summary.", "构建数据集摘要失败。"),
    ("No structures found in this bin.", "该分箱内没有结构。"),
    ("Threshold must be positive.", "阈值必须为正数。"),
    ("Failed to consume force-balance results.", "读取力平衡结果失败。"),
    ("All scanned structures satisfy the net-force threshold.", "所有已扫描结构都满足净力阈值。"),
    (" structures shifted", " 个结构已平移"),
    ("Unmatched examples: ", "未匹配示例："),
    ("unmatched examples: ", "未匹配示例："),
    ("Failed to import baseline preset.", "导入基准预设失败。"),
    ("Please select a preset to export.", "请先选择要导出的预设。"),
    ("Preset not found.", "未找到预设。"),
    ("Failed to export preset.", "导出预设失败。"),
    ("Please select a preset to delete.", "请先选择要删除的预设。"),
    ("Failed to delete preset.", "删除预设失败。"),
    ("Selected preset unavailable.", "选中的预设不可用。"),
    ("Failed to export dataset summary.", "导出数据集摘要失败。"),
    ("Failed to export image.", "导出图片失败。"),
    ("Failed to reset view.", "重置视图失败。"),
    ("Failed to export data.", "导出数据失败。"),
    ("RandomOccupancy: missing composition (Config_type Comp tag or manual input).", "RandomOccupancy：缺少成分信息（Config_type 中的 Comp 标签或手动输入）。"),
    ("No input structure available to test.", "没有可用于测试的输入结构。"),
    ("No atoms selected by 'apply to' settings.", "“应用到”设置没有选中任何原子。"),
    ("SetMagneticMoments: no usable initial_magmoms found.", "SetMagneticMoments：未找到可用的 initial_magmoms。"),
    ("MagneticOrder: AFM mode 'group A/B' requires arrays['group']; falling back to k-vector.", "MagneticOrder：AFM 的“group A/B”模式需要 arrays['group']；已回退到 k-vector。"),
    ("generate descriptors ...", "正在生成描述符……"),
    ("read nep.in file error", "读取 nep.in 文件失败"),
    ("No NEP model file found; the program will use nep89 instead.", "未找到 NEP 模型文件；将临时使用内置 NEP89 模型继续计算。"),
    ("; the program will use nep89 instead.", "；将临时使用内置 NEP89 模型继续计算。"),
    ("The nep calculator fails to calculate the potentials, use the original potentials instead.", "NEP 计算器未能计算势能，已保留原始势能。"),
    ("The nep calculator fails to calculate the polarizability, use the original polarizability instead.", "NEP 计算器未能计算极化率，已保留原始极化率。"),
    ("The nep calculator fails to calculate the dipole, use the original dipole instead.", "NEP 计算器未能计算偶极矩，已保留原始偶极矩。"),
    ("An unknown error occurred while saving. The error message has been output to the log!", "保存时发生未知错误，错误详情已写入日志。"),
    ("Failed to create custom calculator; falling back to NEP.", "创建自定义计算器失败，已回退到 NEP。"),
    ("Failed to import NEP.\n To use the display functionality normally, please prepare the *.out and descriptor.out files.", "导入 NEP 失败。\n如需正常使用显示功能，请准备 *.out 和 descriptor.out 文件。"),
    ("; suggestions were truncated.", "；已截断建议列表。"),
    ("Invalid regex pattern.", "正则表达式无效。"),
    ("Current selection has no points on this plot; FPS will run on full data.", "当前选择在该图中没有点；FPS 将在全部数据上运行。"),
    ("When FPS sampling is performed in the designated area, the program will automatically deselect it, just click to delete!", "在指定区域执行 FPS 采样后，程序会自动取消该区域选择；单击即可删除。"),
    ("No structures were loaded.", "没有加载到结构。"),
    ("load dataset error!", "加载数据集失败！"),
    ("No descriptor data available", "没有可用的描述符数据"),
    ("No selection found; FPS will run on full data.", "没有找到选择集，FPS 将在全部数据上运行。"),
    ("Descriptor dataset is unavailable.", "描述符数据集不可用。"),
    ("Raw descriptors not cached; falling back to reduced space.", "原始描述符未缓存，已回退到降维空间。"),
    ("Edit completed", "编辑完成"),
    ("PCA dimensionality reduction fails", "PCA 降维失败"),
    ("The NEP backend you selected is GPU, but it failed to load on your device; the program has switched to the CPU backend.", "你选择了 GPU NEP 后端，但当前设备加载失败；程序已切换到 CPU 后端。"),
    ("Missing DFT energy; using NEP energy instead.", "缺少 DFT energy，已改用 NEP energy。"),
    ("Missing DFT force; using NEP force instead.", "缺少 DFT force，已改用 NEP force。"),
    ("Missing DFT virial; using NEP virial instead.", "缺少 DFT virial，已改用 NEP virial。"),
    ("Missing DFT stress; using NEP stress instead.", "缺少 DFT stress，已改用 NEP stress。"),
    ("BainPath axis must be x, y, or z.", "BainPath：axis 必须是 x、y 或 z。"),
    ("BainPath mode must be constant_volume, scale_volume, or free_c.", "BainPath：mode 必须是 constant_volume、scale_volume 或 free_c。"),
    ("CellScaling: max_num must be >= 1.", "CellScaling：max_num 必须 >= 1。"),
    ("CellScaling requires three nonzero lattice vectors.", "CellScaling 需要三条非零晶格矢量。"),
    ("Perturb: max_num must be >= 1.", "Perturb：max_num 必须 >= 1。"),
    ("InsertDefect requires at least one host atom.", "插隙与表面吸附至少需要一个宿主原子。"),
    ("InsertDefect requires a finite, non-singular 3x3 cell.", "插隙与表面吸附需要有限且非奇异的 3x3 晶胞。"),
    ("InsertDefect: mode must be 0 (Interstitial) or 1 (Adsorption).", "插隙与表面吸附：模式必须是体相插隙（0）或表面吸附（1）。"),
    ("InsertDefect: structure_count must be an integer.", "插隙与表面吸附：每个输入的输出数必须是整数。"),
    ("InsertDefect: structure_count must be >= 1.", "插隙与表面吸附：每个输入的输出数必须 >= 1。"),
    ("InsertDefect: insert_count must be an integer.", "插隙与表面吸附：每个输出的插入原子数必须是整数。"),
    ("InsertDefect: insert_count must be >= 1.", "插隙与表面吸附：每个输出的插入原子数必须 >= 1。"),
    ("InsertDefect: min_distance must be finite and positive.", "插隙与表面吸附：最小原子间距必须是有限正数。"),
    ("InsertDefect: max_attempts must be an integer.", "插隙与表面吸附：每个原子的尝试次数必须是整数。"),
    ("InsertDefect: max_attempts must be >= 1.", "插隙与表面吸附：每个原子的尝试次数必须 >= 1。"),
    ("InsertDefect: axis must be 0, 1, or 2.", "插隙与表面吸附：表面法向必须是晶格 a、b 或 c。"),
    ("InsertDefect: adsorption height must be finite and positive.", "插隙与表面吸附：吸附高度必须是有限正数。"),
    ("InsertDefect: species must contain at least one element.", "插隙与表面吸附：请至少填写一种插入元素。"),
    ("InsertDefect: seed must be an integer.", "插隙与表面吸附：随机种子必须是整数。"),
    ("InsertDefect: seed must be >= 0.", "插隙与表面吸附：随机种子必须 >= 0。"),
    ("OrganicMolConfig requires at least one atom.", "有机构象采样至少需要一个原子。"),
    ("OrganicMolConfig requires finite Cartesian atom positions.", "有机构象采样需要有限的笛卡尔原子坐标。"),
    ("OrganicMolConfig: pbc_mode must be auto, yes, or no.", "有机构象采样：边界处理模式必须是 auto、yes 或 no。"),
    ("OrganicMolConfig: mixed periodic boundaries are not supported; use pbc_mode=no or provide a fully periodic molecular cell.", "有机构象采样：暂不支持混合周期边界；请选择非周期模式，或提供全三维周期的分子晶胞。"),
    ("OrganicMolConfig: periodic mode requires a finite, nonsingular 3x3 cell.", "有机构象采样：周期模式需要有限且非奇异的 3x3 晶胞。"),
    ("OrganicMolConfig: torsion_range_deg must contain two values.", "有机构象采样：扭转角增量范围必须包含两个值。"),
    ("OrganicMolConfig: torsion_range_deg minimum must not exceed maximum.", "有机构象采样：扭转角增量下限不能大于上限。"),
    ("OrganicMolConfig: bond_keep_max_factor must be >= bond_keep_min_factor.", "有机构象采样：最大键长系数必须不小于最小键长系数。"),
    ("OrganicMolConfig: bo_threshold must be between 0 and 1.", "有机构象采样：键级阈值必须在 0 到 1 之间。"),
    ("OrganicMolConfig: the current settings cannot change any coordinates; enable Gaussian noise or provide an active rotatable bond.", "有机构象采样：当前设置不会改变任何坐标；请启用高斯噪声，或提供至少一条生效的可旋转键。"),
    ("OrganicMolConfig: all requested conformers failed the geometry guards.", "有机构象采样：所有请求构象都未通过几何保护检查。"),
    ("Local Solvation requires at least one host atom.", "局部溶剂壳至少需要一个宿主原子。"),
    ("Local Solvation requires finite Cartesian atom positions.", "局部溶剂壳需要有限的笛卡尔原子坐标。"),
    ("Local Solvation: periodic input requires a finite, nonsingular 3x3 cell.", "局部溶剂壳：周期输入需要有限且非奇异的 3x3 晶胞。"),
    ("Local Solvation: z_range must contain two values.", "局部溶剂壳：笛卡尔 z 范围必须包含两个值。"),
    ("Local Solvation: no center atoms selected.", "局部溶剂壳：没有选中任何溶剂化中心原子。"),
    ("Local Solvation: shell must contain two values.", "局部溶剂壳：备用壳层范围必须包含两个值。"),
    ("Local Solvation: shell outer radius must be larger than inner radius.", "局部溶剂壳：备用壳层外半径必须大于内半径。"),
    ("Local Solvation: box_size must be positive.", "局部溶剂壳：固定盒子边长必须为正数。"),
    ("Local Solvation: flex_torsion_range must contain two values.", "局部溶剂壳：柔性溶剂扭转范围必须包含两个值。"),
    ("Local Solvation: flex_torsion_range minimum must not exceed maximum.", "局部溶剂壳：柔性溶剂扭转范围下限不能大于上限。"),
    ("Solvent Box Fill requires finite Cartesian atom positions.", "周期溶剂盒需要有限的笛卡尔原子坐标。"),
    ("Solvent Box Fill requires a finite, nonsingular input cell.", "周期溶剂盒需要有限且非奇异的输入晶胞。"),
    ("Solvent Box Fill requires periodic boundary conditions.", "周期溶剂盒需要至少开启一个周期方向。"),
    ("Solvent Box Fill: count_mode must be 'fixed' or 'density'.", "周期溶剂盒：目标用量模式必须是固定分子数或名义溶剂密度。"),
    ("Solvent Box Fill: fill_packing must be greater than 0 and at most 1.", "周期溶剂盒：密度计数系数必须大于 0 且不超过 1。"),
    ("Solvent Box Fill: sampling_mode must be one of auto, general, water, loose, dense.", "周期溶剂盒：碰撞配置无效。"),
    ("Solvent Box Fill: flex_torsion_range must contain two values.", "周期溶剂盒：柔性溶剂扭转范围必须包含两个值。"),
    ("Solvent Box Fill: flex_torsion_range minimum must not exceed maximum.", "周期溶剂盒：柔性溶剂扭转范围下限不能大于上限。"),
    ("LayerCopy: dz expression is empty.", "LayerCopy：dz 表达式为空。"),
    ("LayerCopy: no atoms selected by apply settings.", "LayerCopy：“应用到”设置没有选中任何原子。"),
    ("LayerCopy requires at least one atom.", "分层堆叠至少需要一个原子。"),
    ("LayerCopy requires finite Cartesian atom positions.", "分层堆叠需要有限的笛卡尔原子坐标。"),
    ("LayerCopy requires a finite 3x3 cell.", "分层堆叠需要有限的 3x3 晶胞。"),
    ("LayerCopy: layer translation must be positive when layers > 1.", "分层堆叠：总层数大于 1 时，副本平移量必须为正数。"),
    ("LayerCopy: z_range must contain two values.", "分层堆叠：笛卡尔 z 范围必须包含两个值。"),
    ("LayerCopy: wrapping requires a nonsingular cell.", "分层堆叠：坐标折回需要非奇异晶胞。"),
    ("FPS Filter: backend must be auto, cpu, or cuda.", "FPS 代表性采样：后端必须是 auto、cpu 或 cuda。"),
    ("FPS Filter: existing training dataset contains an empty structure.", "FPS 代表性采样：已有训练集中包含空结构。"),
    ("FPS Filter: candidate and existing descriptor dimensions differ.", "FPS 代表性采样：候选集与已有训练集的描述符维度不同。"),
    ("Geometry Filter: minimum volume/atom must not exceed maximum volume/atom.", "几何健全性过滤：最小单原子体积不能超过最大单原子体积。"),
    ("Geometry Filter: minimum density must not exceed maximum density.", "几何健全性过滤：最小密度不能超过最大密度。"),
    ("VibrationModePerturb: amplitude must be positive.", "VibrationModePerturb：amplitude 必须为正数。"),
    ("VibrationModePerturb: modes_per_sample must be >= 1.", "VibrationModePerturb：modes_per_sample 必须 >= 1。"),
    ("GroupLabel: structure has no valid cell.", "GroupLabel：结构没有有效晶胞。"),
    ("Random Packing: structures must be >= 1.", "Random Packing：structures 必须 >= 1。"),
    ("Random Packing: min_distance must be positive.", "Random Packing：min_distance 必须为正数。"),
    ("Random Packing: max_attempts_per_atom must be >= 1.", "Random Packing：max_attempts_per_atom 必须 >= 1。"),
    ("Random Packing requires a non-singular input cell.", "Random Packing 需要非奇异的输入晶胞。"),
    ("Spin Disorder requires vector magnetic moments or liftable scalar magmoms.", "Spin Disorder 需要矢量磁矩，或可提升为矢量的标量 magmoms。"),
    ("Spin Disorder found no eligible nonzero magnetic moments.", "Spin Disorder 未找到符合条件的非零磁矩。"),
    ("Spin Disorder requires at least one positive disorder fraction.", "Spin Disorder 至少需要一个正的无序比例。"),
    ("Spin Disorder did not generate any structures.", "Spin Disorder 没有生成任何结构。"),
    ("Card JSON must be an object, a list, or an exported workflow.", "卡片 JSON 必须是对象、列表或导出的工作流。"),
    ("Card JSON does not contain any cards.", "卡片 JSON 中没有任何卡片。"),
    ("Each card JSON entry must be an object.", "每个卡片 JSON 条目都必须是对象。"),
    ("Each card JSON entry must contain a class name.", "每个卡片 JSON 条目都必须包含类名。"),
    (" (use Sobol for order>=4)", "（order >= 4 时请使用 Sobol）"),
)


_RUNTIME_EXCEPTION_REPLACEMENTS = (
    ("OrganicMolConfig:", "有机构象采样："),
    ("Local Solvation:", "局部溶剂壳："),
    ("Solvent Box Fill:", "周期溶剂盒："),
    ("LayerCopy:", "分层堆叠："),
    ("FPS Filter:", "FPS 代表性采样："),
    ("Geometry Filter:", "几何健全性过滤："),
    ("must be >= 1", "必须 >= 1"),
    ("must be positive", "必须为正数"),
    ("must be finite", "必须是有限值"),
    ("must be non-negative", "必须为非负数"),
    ("must contain exactly three values: start, stop, step", "必须包含三个值：起点、终点、步长"),
    ("step must be positive", "步长必须为正数"),
    ("values must be finite", "取值必须是有限值"),
    ("requires at least one atom", "至少需要一个原子"),
    ("requires at least two atoms", "至少需要两个原子"),
    ("requires a nonsingular 3x3 cell", "需要非奇异的 3x3 晶胞"),
    ("requires a non-singular input cell", "需要非奇异的输入晶胞"),
    ("produced a singular gamma angle", "生成了奇异的 gamma 角"),
    ("unsupported mode", "不支持的模式"),
    ("unsupported", "不支持"),
    ("invalid", "无效"),
    ("Invalid", "无效"),
    ("Unknown", "未知"),
    ("requires", "需要"),
    ("failed", "失败"),
    ("Failed", "失败"),
    ("empty", "为空"),
    ("no atoms selected", "没有选中原子"),
    ("not found", "未找到"),
    ("does not contain", "不包含"),
    ("shape mismatch", "形状不匹配"),
    ("is not allowed", "不允许"),
    ("NaN/Inf", "NaN/Inf"),
    ("Missing DFT ", "缺少 DFT "),
    ("; using NEP ", "；已改用 NEP "),
    (" instead.", "。"),
    (" already exists, please delete it first", " 已存在，请先删除它"),
    (" already exists!", " 已存在！"),
    (" does not exist!", " 不存在！"),
    (" structures with |ΣF| > ", " 个结构满足 |ΣF| > "),
    ("The index is ", "索引为 "),
    (" must be ", " 必须是 "),
    ("an integer", "整数"),
    ("a finite number", "有限数值"),
    (" or ", " 或 "),
)


def translate_runtime_message(message) -> str:
    """Translate late-bound UI messages and common runtime errors."""
    if isinstance(message, CardOperationError):
        template = QCoreApplication.translate("CardOperationError", message.template)
        values = dict(message.values)
        if isinstance(values.get("field"), str):
            values["field"] = QCoreApplication.translate(
                "CardOperationField", values["field"]
            )
        return template.format(**values)
    text = str(message)
    translated = QCoreApplication.translate("RuntimeMessage", text)
    if translated != text:
        return translated
    if not _zh_runtime_messages_enabled():
        return text

    for prefix, replacement in _RUNTIME_PREFIX_REPLACEMENTS:
        if text.startswith(prefix):
            text = replacement + text[len(prefix):]
            break
    for source, replacement in _RUNTIME_TEXT_REPLACEMENTS:
        text = text.replace(source, replacement)
    for source, replacement in _RUNTIME_EXCEPTION_REPLACEMENTS:
        text = text.replace(source, replacement)
    return text


def _runtime_message_catalog() -> None:
    """Literal catalog for lupdate; runtime translations are applied centrally."""
    QCoreApplication.translate("RuntimeMessage", "__language_probe__")
    for text, _replacement in (
        *_RUNTIME_PREFIX_REPLACEMENTS,
        *_RUNTIME_TEXT_REPLACEMENTS,
        *_RUNTIME_EXCEPTION_REPLACEMENTS,
    ):
        QCoreApplication.translate("RuntimeMessage", text)


def _card_operation_error_catalog() -> None:
    """Literal catalog for structured card errors discovered by lupdate."""
    QCoreApplication.translate(
        "CardOperationError",
        "Maximum output {budget} is smaller than the {groups} element sets. "
        "Increase the output limit or remove unneeded systems.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Perturb: Sobol sampling supports at most {max_atoms} atoms; "
        "use Uniform sampling for larger structures.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Perturb: {element} is not a valid element symbol for a displacement limit.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Perturb: element {element} has more than one displacement limit; keep only one row.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Perturb: the displacement limit for {element} must be finite and non-negative.",
    )
    QCoreApplication.translate("CardOperationError", "Unsupported crystal prototype: {lattice}.")
    QCoreApplication.translate("CardOperationError", "Maximum outputs must be at least 1.")
    QCoreApplication.translate(
        "CardOperationError", "The hcp c/a ratio must be a positive finite number."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Enter one valid chemical element symbol, for example Cu, Fe, or Mg.",
    )
    QCoreApplication.translate(
        "CardOperationError", "The lattice-constant range must contain finite numbers."
    )
    QCoreApplication.translate("CardOperationError", "Lattice constants must be positive.")
    QCoreApplication.translate(
        "CardOperationError",
        "Lattice perturbation produced an invalid or singular cell. Reduce the maximum relative change.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Lattice strain values must be greater than -100%."
    )
    QCoreApplication.translate(
        "CardOperationError", "Select one or more unique lattice axes: a, b, or c."
    )
    QCoreApplication.translate(
        "CardOperationError", "The lattice-constant step must be positive."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Enter one element symbol or the X placeholder for every visible sublattice.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Invalid element or placeholder {element}; use a chemical element symbol or X.",
    )
    QCoreApplication.translate(
        "CardOperationError", "The mode coefficient scale must be a positive finite number."
    )
    QCoreApplication.translate(
        "CardOperationError", "Modes combined per sample must be at least 1."
    )
    QCoreApplication.translate("CardOperationError", "Structures per input must be an integer.")
    QCoreApplication.translate("CardOperationError", "Structures per input must be at least 1.")
    QCoreApplication.translate(
        "CardOperationError", "Coefficient distribution must be Normal or Uniform."
    )
    QCoreApplication.translate(
        "CardOperationError", "The absolute frequency cutoff must be a finite non-negative number."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vibrational perturbation needs at least one usable mode on every input structure.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Finite frequencies are required when frequency filtering or scaling is enabled.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Frequency weighting requires non-zero frequencies for every usable mode.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Modes per sample is {requested}, but only {available} usable modes are available.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Maximum perturbation angle must be between 0 and 180 degrees."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Spin Perturb requires spin or initial magnetic moments on every input structure.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Scalar magnetic moments must be lifted to vectors before they can be rotated.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The scalar lift direction must contain three finite Cartesian components.",
    )
    QCoreApplication.translate(
        "CardOperationError", "The scalar lift direction must be non-zero."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Spin Perturb could not convert the input magnetic moments to vectors.",
    )
    QCoreApplication.translate("CardOperationError", "Structures per input must be a positive integer.")
    QCoreApplication.translate("CardOperationError", "Maximum attempts per atom must be a positive integer.")
    QCoreApplication.translate("CardOperationError", "Generated atom budget per input must be a positive integer.")
    QCoreApplication.translate("CardOperationError", "Random seed must be a non-negative integer.")
    QCoreApplication.translate(
        "CardOperationError", "Global minimum distance must be a positive finite number."
    )
    QCoreApplication.translate(
        "CardOperationError", "Random Packing requires a finite, non-singular input cell."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Composition mode must be Use input composition or Manual atom counts.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Random Packing needs at least one atom. Load a non-empty input or enter a manual composition.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Requested outputs contain {requested} generated atoms per input, exceeding the budget of {budget}. Reduce structures or atom counts, or raise the budget deliberately.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Unknown chemical element {element} in the composition."
    )
    QCoreApplication.translate(
        "CardOperationError", "Invalid composition item {item}; use Element:count, for example Fe:32."
    )
    QCoreApplication.translate(
        "CardOperationError", "Atom count for {element} must be a positive integer."
    )
    QCoreApplication.translate(
        "CardOperationError", "Invalid pair-distance rule {item}; use A-B:value, for example Fe-O:1.8."
    )
    QCoreApplication.translate(
        "CardOperationError", "Unknown chemical element {element} in a pair-distance rule."
    )
    QCoreApplication.translate(
        "CardOperationError", "Minimum distance for {left}-{right} must be a positive finite number."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Random Packing could not place {element} after {attempts} attempts. Reduce the minimum distances, enlarge the cell, or lower the atom count.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Random Packing could not generate any output after {failures} failed attempts. Reduce the minimum distances, enlarge the cell, or lower the atom count.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Magnitude scale bounds must be finite and non-negative."
    )
    QCoreApplication.translate(
        "CardOperationError", "Magnitude scale minimum must not exceed the maximum."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Increase the perturbation angle or enable a magnitude scale range that changes the moments.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Exact correlated sampling supports at most {maximum} eligible non-zero moments; "
        "the current selection has {actual}. Reduce the target elements or use a smaller structure.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Spin Spiral maximum outputs must be at least 1."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "No lattice-compatible spin-spiral period exists in the requested range. "
        "For a period of {period} Å, try a {multipliers} supercell.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "No lattice-compatible spin-spiral period exists in the requested range. "
        "Change the period range or expand the cell along the propagation axis.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Spin Spiral requires at least one non-zero magnetic moment. "
        "Add moments upstream or select the element-map source and enter a non-zero magnitude.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Maximum outputs must be at least {minimum} for the selected folded-helix sequence mode.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Folded Helix requires at least one non-zero magnetic moment. "
        "Add moments upstream or select the element-map source and enter a non-zero magnitude.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Folded Helix needs at least 3 detected layers along the layer axis; "
        "the current settings detect {actual}. Check the layer axis and tolerance.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Layer Groups needs at least two detected atomic layers; the current settings detect {actual}. "
        "Expand the cell, choose another plane, or reduce the layer tolerance.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Layer Groups needs a finite, non-singular 3D cell."
    )
    QCoreApplication.translate(
        "CardOperationError", "Layer tolerance must be a positive finite distance."
    )
    QCoreApplication.translate(
        "CardOperationError", "Plane index must be 100, 010, 001, 110, or 111."
    )
    QCoreApplication.translate(
        "CardOperationError", "Layer group labels must be non-empty."
    )
    QCoreApplication.translate(
        "CardOperationError", "Layer group A and B labels must be different."
    )
    QCoreApplication.translate(
        "CardOperationError", "Targeted Vacancy requires at least one vacancy rule."
    )
    QCoreApplication.translate(
        "CardOperationError", "Targeted Vacancy requires at least two atoms."
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancy rule {rule} is invalid."
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancy rule {rule} requires an element."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} needs one integer count or an ordered minimum/maximum pair.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancy rule {rule} count must be at least 0."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} count mode must be Fixed count or Random range.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} fixed count must use the same minimum and maximum.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancy rule {rule} fixed count must be at least 1."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} random range must allow at least one vacancy.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} group must contain at least one non-empty label.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} requests group labels, but the input structure has no group array.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancy rule {rule} matched no atoms ({target})."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy rule {rule} requests up to {requested} vacancies, but only {available} atoms match ({target}).",
    )
    QCoreApplication.translate(
        "CardOperationError", "Maximum outputs per input must be at least 1."
    )
    QCoreApplication.translate(
        "CardOperationError", "Random seed must be at least 0."
    )
    QCoreApplication.translate(
        "CardOperationError", "{field} must be an integer."
    )
    QCoreApplication.translate(
        "CardOperationError", "{field} must be at least {minimum}."
    )
    QCoreApplication.translate(
        "CardOperationError", "{field} must be a finite number."
    )
    QCoreApplication.translate(
        "CardOperationError", "{field} must be positive."
    )
    QCoreApplication.translate(
        "CardOperationError", "Molecular Conformers requires at least one atom."
    )
    QCoreApplication.translate(
        "CardOperationError", "Molecular Conformers requires finite Cartesian atom positions."
    )
    QCoreApplication.translate(
        "CardOperationError", "Output boundary must be Follow input, 3D periodic, or Nonperiodic."
    )
    QCoreApplication.translate(
        "CardOperationError", "Follow input does not support mixed periodic boundaries; choose Nonperiodic or provide full 3D PBC."
    )
    QCoreApplication.translate(
        "CardOperationError", "3D periodic mode requires a finite, nonsingular 3×3 cell."
    )
    QCoreApplication.translate(
        "CardOperationError", "Torsion increment range must contain a minimum and maximum."
    )
    QCoreApplication.translate(
        "CardOperationError", "Torsion increment minimum must not exceed its maximum."
    )
    QCoreApplication.translate(
        "CardOperationError", "Maximum bond length must not be smaller than minimum bond length."
    )
    QCoreApplication.translate(
        "CardOperationError", "Bond-order threshold must be between 0 and 1."
    )
    QCoreApplication.translate(
        "CardOperationError", "The current settings cannot change coordinates; add coordinate noise or provide an active rotatable bond."
    )
    QCoreApplication.translate(
        "CardOperationError", "All requested conformers failed the geometry guards; narrow the torsion range, reduce coordinate noise, or inspect the distance limits."
    )
    QCoreApplication.translate("CardOperationError", "Solvent Shell requires at least one host atom.")
    QCoreApplication.translate("CardOperationError", "Solvent Shell requires finite Cartesian atom positions.")
    QCoreApplication.translate("CardOperationError", "Periodic input requires a finite, nonsingular 3×3 cell.")
    QCoreApplication.translate("CardOperationError", "Solvent XYZ must describe one valid non-empty molecule.")
    QCoreApplication.translate("CardOperationError", "Cartesian z range must contain a minimum and maximum.")
    QCoreApplication.translate("CardOperationError", "The center selection does not match any host atoms.")
    QCoreApplication.translate("CardOperationError", "Fallback center-to-COM shell must contain an inner and outer radius.")
    QCoreApplication.translate("CardOperationError", "Fallback shell outer radius must be larger than its inner radius.")
    QCoreApplication.translate("CardOperationError", "Fixed box size must be positive.")
    QCoreApplication.translate("CardOperationError", "Flexible torsion range must contain a minimum and maximum.")
    QCoreApplication.translate("CardOperationError", "Flexible torsion minimum must not exceed its maximum.")
    QCoreApplication.translate("CardOperationError", "No solvent molecule could be placed; adjust the centers, shell, collision rule, or free volume.")
    QCoreApplication.translate("CardOperationError", "No solvent molecule could be placed; adjust the centers, shell, collision rule, or free volume. The periodic structure appears to have no solvent-accessible void; use a structure with free volume or a larger periodic cell.")
    QCoreApplication.translate("CardOperationError", "Only {placed} of {requested} solvent molecules could be placed.")
    QCoreApplication.translate("CardOperationError", "Unsupported placement method: {mode}.")
    QCoreApplication.translate("CardOperationError", "Supported ion hydration requires a water solvent molecule.")
    QCoreApplication.translate("CardOperationError", "Supported ion hydration requires every selected center to be Li, Na, K, Mg, Ca, Sr, Ba, or Zn.")
    QCoreApplication.translate("CardOperationError", "Enter at least one center element.")
    QCoreApplication.translate("CardOperationError", "Unsupported center selection: {mode}.")
    QCoreApplication.translate("CardOperationError", "Enter at least one 1-based center index.")
    QCoreApplication.translate("CardOperationError", "Center index {index} is outside the valid range 1–{natoms}.")
    QCoreApplication.translate("CardOperationError", "Center indices must use 1-based integers and ranges such as 1,3,5-8.")
    QCoreApplication.translate(
        "CardOperationError",
        "Targeted Vacancy could not generate a valid non-empty structure. "
        "Reduce overlapping rule counts, broaden the groups, or expand the structure.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Global Vacancy requires at least two atoms."
    )
    QCoreApplication.translate(
        "CardOperationError", "Site sampling must be Uniform or Sobol."
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancies per output must be Fixed or Variable."
    )
    QCoreApplication.translate(
        "CardOperationError", "Maximum outputs per input must be a positive integer."
    )
    QCoreApplication.translate("CardOperationError", "Vacancies must be an integer.")
    QCoreApplication.translate("CardOperationError", "Vacancies must be at least 1.")
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancies must be at most {maximum} for this input so at least one atom remains.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Vacancy fraction must be greater than 0 and less than 1."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Vacancy fraction is too small for this input; use at least {minimum} to remove one atom.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Sobol sampling supports at most {maximum} atoms; use Uniform for larger inputs.",
    )
    QCoreApplication.translate("CardOperationError", "Random seed must be an integer.")
    QCoreApplication.translate("CardOperationError", "Random seed must be at least 0.")
    QCoreApplication.translate(
        "CardOperationError", "GSFE Path requires at least one atom."
    )
    QCoreApplication.translate(
        "CardOperationError", "GSFE Path requires a finite, nonsingular 3×3 cell."
    )
    QCoreApplication.translate(
        "CardOperationError", "The fault-plane indices must not all be zero."
    )
    QCoreApplication.translate(
        "CardOperationError", "The in-plane direction must not be zero."
    )
    QCoreApplication.translate(
        "CardOperationError", "The in-plane indices produce a zero shift vector."
    )
    QCoreApplication.translate(
        "CardOperationError", "The shift direction must lie in the fault plane."
    )
    QCoreApplication.translate(
        "CardOperationError", "Displacement unit must be Vector fraction or Å distance."
    )
    QCoreApplication.translate(
        "CardOperationError", "GSFE Path requires finite Cartesian atom positions."
    )
    QCoreApplication.translate(
        "CardOperationError", "Displacement path needs a start, end, and positive step."
    )
    QCoreApplication.translate(
        "CardOperationError", "{label} must contain exactly three integers."
    )
    QCoreApplication.translate(
        "CardOperationError", "The fault-plane indices produce a zero normal."
    )
    QCoreApplication.translate(
        "CardOperationError", "GSFE Path requires a nonzero third cell vector."
    )
    QCoreApplication.translate(
        "CardOperationError", "The third cell vector must be normal to the current ab fault plane."
    )
    QCoreApplication.translate(
        "CardOperationError", "GSFE Path requires finite projected atom coordinates."
    )
    QCoreApplication.translate(
        "CardOperationError", "GSFE Path requires atoms on at least two distinct layers."
    )
    QCoreApplication.translate(
        "CardOperationError", "Thickness fraction must be between 0 and 1."
    )
    QCoreApplication.translate(
        "CardOperationError", "Lower layer index must select a layer below the top layer."
    )
    QCoreApplication.translate(
        "CardOperationError", "Cut position must be Middle, Thickness, or Layer index."
    )
    QCoreApplication.translate(
        "CardOperationError", "The cut must leave atoms on both sides; adjust its position."
    )
    QCoreApplication.translate(
        "CardOperationError", "{label} must be an integer."
    )
    QCoreApplication.translate(
        "CardOperationError", "Invalid element symbols in the target list: {elements}."
    )
    QCoreApplication.translate(
        "CardOperationError", "No non-zero magnetic moments match the selected elements."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Magnetic response needs vector spin or initial magnetic moments on the input structure.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Magnetic response needs finite magnetic moments with at least one non-zero vector.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Maximum structures is smaller than the coordinate count of one complete response group.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "No non-zero magnetic moments match the selected atoms and elements.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The input needs a non-zero magnetic moment in both group '{group_a}' and group '{group_b}'. "
        "Check the group labels or add Layer Groups upstream.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "No atom pairs match the selected neighbor shell and automatic-pair filters.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "No valid magnetic atom pairs match the left and right indices.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Maximum structures must be at least {required} for the selected texture response path.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Cell-reciprocal q needs a finite, non-singular 3D cell."
    )
    QCoreApplication.translate(
        "CardOperationError", "The reciprocal-cell index must contain three finite integers."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The reciprocal-cell index cannot be (0, 0, 0) for a spiral response.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "q definition must be Cell reciprocal vector or Cartesian vector.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The Cartesian base q vector must be non-zero for a spiral response.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "q does not close across periodic cell vector {index}. Use the cell-reciprocal q mode, "
        "or change q and the supercell together.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The two symmetric-shear directions must be perpendicular Cartesian vectors.",
    )
    QCoreApplication.translate(
        "CardOperationError", "The Bain lattice axis must be a, b, or c."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Angle strain produced an invalid or singular cell. Reduce the angle increments.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian shear produced an invalid or singular cell. Reduce the shear components.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Composition Gradient requires at least two equal-count groups.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Composition Gradient requires at least one random sample.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Composition Gradient requires at least two eligible sites.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "List one or more existing elements for the selected site scope.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Composition Space Sampling requires at least two valid elements.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Composition Space Sampling has unknown element symbol(s): {elements}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "None of the selected component counts is feasible for {count} elements.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Maximum target compositions per input must be at least 1.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Grid sampling for four or five components requires a step of 1/n, such as 0.1 or 0.05.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The current composition constraints produce no target compositions.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The current settings produce no unique target compositions.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Maximum target compositions per input cannot exceed {maximum}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The Grid settings require about {count} simplex points before budgeting. Increase the step or use Sobol; the safe limit is {maximum}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Invalid Cartesian position filter syntax: {reason}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian position filter contains unsupported syntax.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian position filter may use only x, y, and z; unknown name(s): {names}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian position filter may use only finite numeric constants.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian position filter must be a comparison or a boolean expression.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian position filter divides by zero for at least one atom.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian position filter produces non-finite arithmetic for at least one atom.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Enter one valid target element symbol, such as O, Si, or Fe.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Unknown target element symbol: {element}."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Replacement ratios must be finite and non-negative.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Add at least one replacement element with a positive relative ratio.",
    )
    QCoreApplication.translate(
        "CardOperationError", "Replacement element {element} appears more than once."
    )
    QCoreApplication.translate(
        "CardOperationError", "Unknown replacement element symbol(s): {elements}."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Replacement elements must not include the target element "
        "{element}; use Random Doping for partial replacement.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Element allocation must be Independent random assignment or Match overall ratio.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Conditional Replace seed must be a non-negative integer.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Conditional Replace requires finite Cartesian atom positions.",
    )
    QCoreApplication.translate(
        "CardOperationError", "The input structure contains no {element} atoms."
    )
    QCoreApplication.translate(
        "CardOperationError",
        "The Cartesian position filter matches no {element} atoms.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Layer spacing must use Surface gap or Copy translation.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Total layers must be an integer of at least {minimum}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Warp selection must be All atoms, Selected elements, or Cartesian z range.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Atom budget per output must be an integer of at least {minimum}.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Layer spacing must be a finite non-negative distance.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Additional top vacuum must be a finite non-negative distance.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Cartesian z range must contain two finite distances.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Copy translation must be positive when total layers is greater than 1.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Layer Stack would create {actual} atoms per output, above the {limit}-atom "
        "budget. Reduce the layer count or increase the budget.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Copy translation {translation} Å is smaller than the warped slab thickness "
        "{thickness} Å, giving a negative surface gap {gap} Å.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Layer Stack needs a finite, non-singular 3D cell.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Extending the cell requires lattice vector c to have a positive Cartesian z component.",
    )
    QCoreApplication.translate(
        "CardOperationError",
        "Layer Stack would create a singular or inverted final cell.",
    )


def _card_operation_field_catalog() -> None:
    """Literal field names interpolated into structured card errors."""
    QCoreApplication.translate("CardOperationField", "Maximum outputs per input")
    QCoreApplication.translate("CardOperationField", "Bonds rotated per output")
    QCoreApplication.translate("CardOperationField", "Large-molecule threshold")
    QCoreApplication.translate("CardOperationField", "Local subtree cap")
    QCoreApplication.translate("CardOperationField", "Retries per output")
    QCoreApplication.translate("CardOperationField", "Coordinate noise")
    QCoreApplication.translate("CardOperationField", "Torsion increment range")
    QCoreApplication.translate("CardOperationField", "Bond detection radius")
    QCoreApplication.translate("CardOperationField", "Minimum bond length")
    QCoreApplication.translate("CardOperationField", "Maximum bond length")
    QCoreApplication.translate("CardOperationField", "Minimum nonbonded distance")
    QCoreApplication.translate("CardOperationField", "Short-bond rotation cutoff")
    QCoreApplication.translate("CardOperationField", "Nonperiodic display box")
    QCoreApplication.translate("CardOperationField", "Pauling decay length")
    QCoreApplication.translate("CardOperationField", "Bond-order threshold")
    QCoreApplication.translate("CardOperationField", "Random seed")
    QCoreApplication.translate("CardOperationField", "Independent outputs per input")
    QCoreApplication.translate("CardOperationField", "Total solvent molecules per output")
    QCoreApplication.translate("CardOperationField", "Placement attempts per output")
    QCoreApplication.translate("CardOperationField", "Cartesian z range")
    QCoreApplication.translate("CardOperationField", "Fallback center-to-COM shell")
    QCoreApplication.translate("CardOperationField", "Uniform minimum distance")
    QCoreApplication.translate("CardOperationField", "Element-radius collision scale")
    QCoreApplication.translate("CardOperationField", "Fixed box size")
    QCoreApplication.translate("CardOperationField", "Auto-box padding")
    QCoreApplication.translate("CardOperationField", "Minimum auto-box edge")
    QCoreApplication.translate("CardOperationField", "Flexible conformer pool")
    QCoreApplication.translate("CardOperationField", "Flexible torsions per conformer")
    QCoreApplication.translate("CardOperationField", "Flexible conformer noise")
    QCoreApplication.translate("CardOperationField", "Flexible torsion range")


class MessageManager(QObject):
    """Qt message center singleton for showing InfoBars and message boxes.

    Typical usage:
        from NepTrainKit.ui.messages import MessageManager
        MessageManager.send_info_message("Hello")
    """

    _instance = None
    showMessageSignal = Signal(InfoBarIcon, str, str)
    showBoxSignal = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__()
        self._parent = parent
        self._instance: MessageManager
        self.showMessageSignal.connect(self._show_message)
        self.showBoxSignal.connect(self._show_box)

    @classmethod
    def _createInstance(cls, parent=None):
        if not cls._instance:
            cls._instance = MessageManager(parent)
        from NepTrainKit.core.message import MessageManager as CoreMessageManager

        CoreMessageManager.register_sink(cls)

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classmethod
    def send_info_message(cls, message, title=None):
        message = translate_runtime_message(message)
        title = _tr("Tip") if title is None else title
        title = translate_runtime_message(title)
        if cls._instance is None:
            logger.info(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.INFORMATION, message, title)

    @classmethod
    def send_success_message(cls, message, title=None):
        message = translate_runtime_message(message)
        title = _tr("Success") if title is None else title
        title = translate_runtime_message(title)
        if cls._instance is None:
            logger.success(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.SUCCESS, message, title)

    @classmethod
    def send_warning_message(cls, message, title=None):
        message = translate_runtime_message(message)
        title = _tr("Warning") if title is None else title
        title = translate_runtime_message(title)
        if cls._instance is None:
            logger.warning(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.WARNING, message, title)

    @classmethod
    def send_error_message(cls, message, title=None):
        message = translate_runtime_message(message)
        title = _tr("Error") if title is None else title
        title = translate_runtime_message(title)
        if cls._instance is None:
            logger.error(message)
        else:
            cls._instance.showMessageSignal.emit(InfoBarIcon.ERROR, message, title)

    @classmethod
    def send_message_box(cls, message, title=None):
        message = translate_runtime_message(message)
        title = _tr("Tip") if title is None else title
        title = translate_runtime_message(title)
        if cls._instance is None:
            logger.info(message)
        else:
            cls._instance.showBoxSignal.emit(message, title)

    def _show_box(self, message, title):
        w = MessageBox(title, message, self._parent)
        w.cancelButton.hide()
        w.exec_()

    def _show_message(self, msg_type, msg, title):
        if msg_type == InfoBarIcon.ERROR:
            duration = 10000
        elif msg_type == InfoBarIcon.WARNING:
            duration = 8000
        else:
            duration = 5000
        InfoBar.new(
            msg_type,
            title=title,
            content=msg,
            orient=Qt.Orientation.Vertical,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=duration,
            parent=self._parent,
        )
