"""Screenshot scenario registry for documentation assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_WINDOW_SIZE = (1200, 700)
DEFAULT_OUTPUT_DIR = Path("docs/source/_static/image/generated")

ZH_TEXT = {
    "NEP Dataset Display overview": "NEP 数据集展示总览",
    "Open data": "打开数据",
    "Error plots": "误差图",
    "Structure viewer": "结构查看器",
    "Search and selection": "搜索与选择",
    "Training Set Audit overview": "训练集评估总览",
    "Phase evidence on the composition map": "组分地图中的相结构证据",
    "Magnetic-type shares": "磁类型占比",
    "Make Data workspace": "Make Data 工作区",
    "Open input structures": "打开输入结构",
    "Add new card": "添加卡片",
    "Run selected cards": "运行选中卡片",
    "Workflow workspace": "流程工作区",
    "Make Data lattice strain quickstart": "Make Data 晶格应变快速上手",
    "Add Lattice Strain card": "添加 Lattice Strain 卡片",
    "Set strain range": "设置应变范围",
    "Export generated structures": "导出生成结构",
    "Energy baseline shift entry": "能量基线平移入口",
    "Confirm a large constant energy baseline": "确认存在巨大常数基线",
    "Open energy baseline shift": "打开能量基线平移",
    "Energy plot after baseline shift": "平移后的能量图",
    "The large constant baseline is removed": "巨大常数基线已去除",
    "Farthest point sampling": "最远点采样",
    "Inspect diversity in descriptor space": "先看描述符空间的多样性",
    "Open farthest point sampling": "打开最远点采样",
    "FPS representatives": "FPS 代表结构",
    "Eight representatives are selected": "已选中 8 个代表结构",
    "Maximum-error review": "最大误差结构复核",
    "Activate the energy plot": "先激活能量图",
    "Find maximum-error structures": "查找最大误差结构",
    "Selected structures need review": "选中结构需要逐个复核",
    "Inspect structure and Config_type": "查看结构和 Config_type",
    "DFT-D3 correction entry": "DFT-D3 校正入口",
    "Confirm the current reference labels": "确认当前参考标签口径",
    "Open DFT-D3 correction": "打开 DFT-D3 校正",
    "Energy plot after adding DFT-D3": "加上 DFT-D3 后的能量图",
    "Reference labels now include DFT-D3": "参考标签现已包含 DFT-D3",
    "Select structures by index": "按索引选取结构",
    "Open index selection": "打开索引选择",
    "Five requested structures are selected": "已选中指定的 5 个结构",
    "Structure quality checks": "结构质量检查",
    "Scan suspicious distances": "检查异常原子距离",
    "Scan non-zero net forces": "检查非零净力",
    "Two deliberately corrupted frames are selected": "两个故意破坏的示例结构已被选中",
    "Edit selected metadata": "修改选中结构的元数据",
    "Select the structures to edit": "先选中要修改的结构",
    "Open the metadata editor": "打开元数据编辑器",
    "Config_type is updated for the selection": "选中结构的 Config_type 已更新",
    "Dataset summary from real data": "真实数据集概览",
    "Force distribution from real data": "真实数据的力分布",
    "Shared card controls": "卡片通用按钮",
    "Enable or skip this card": "启用或跳过这张卡片",
    "Collapse or expand settings": "折叠或展开参数",
    "Open this card's manual": "打开这张卡片的手册",
    "Show card provenance": "查看卡片来源信息",
    "Copy this card as JSON": "复制单张卡片 JSON",
    "View this card's output": "查看这张卡片的输出",
    "Export this card's output": "导出这张卡片的输出",
    "Remove this card": "移除这张卡片",
    "A complete card workflow": "一条完整的卡片流程",
    "Add or search for cards": "添加或搜索卡片",
    "Copy the whole workflow as JSON": "用 JSON 复制整条流程",
    "Top-level cards run in order": "顶层卡片按顺序串行",
    "Grouped cards branch from one input": "卡片组从同一输入分支",
    "Run all enabled cards": "运行所有已启用卡片",
    "Verified workflow result": "已验证的流程结果",
    "Two branches produce four structures": "两条分支共生成 4 个结构",
    "The final filter keeps eight structures": "最终筛选保留 8 个结构",
    "View all enabled card outputs": "查看所有已启用卡片的输出",
    "Inspect the final card output": "单独检查最终卡片输出",
    "Export the final card output": "导出最终卡片输出",
}

EN_TEXT = {
    "先读当前结论": "Read the conclusion first",
    "确认数据概况": "Confirm the dataset summary",
    "按建议顺序复核": "Review in the recommended order",
    "导出分层报告": "Export the layered report",
    "切换元素与证据层": "Switch element and evidence layer",
    "相分布随成分变化": "Phase distribution across composition",
    "精确成分与结构数": "Exact composition and structure count",
    "筛选相并回看结构": "Filter phases and inspect structures",
    "选择磁类型证据": "Select magnetic-type evidence",
    "切换三种占比关系": "Switch among three share views",
    "按结构帧查看占比": "Inspect shares by structure frame",
    "点击色块回看结构": "Click a segment to inspect structures",
}


def localized_text(text: str, language: str) -> str:
    """Return screenshot annotation text for a runtime language."""
    if language == "zh_CN":
        return ZH_TEXT.get(text, text)
    if language == "en_US":
        return EN_TEXT.get(text, text)
    raise ValueError(f"Unsupported screenshot language: {language}")


@dataclass(frozen=True)
class Annotation:
    """A visual callout drawn on top of a captured UI screenshot."""

    number: str
    label: str
    target: str | tuple[int, int, int, int]
    badge: str | tuple[int, int] | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative metadata for a documentation screenshot."""

    name: str
    title: str
    runner: str
    output: Path
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE
    annotations: tuple[Annotation, ...] = ()
    description: str = ""
    options: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, ScenarioSpec] = {
    "show_nep_overview": ScenarioSpec(
        name="show_nep_overview",
        title="NEP Dataset Display overview",
        runner="show_nep_overview",
        output=DEFAULT_OUTPUT_DIR / "show_nep_overview.png",
        description="Main NEP Dataset Display window with top-level regions annotated.",
        annotations=(
            Annotation("1", "Open data", "widget:open_dir_button", "right"),
            Annotation("2", "Error plots", "widget:show_nep_interface.plot_widget", "top-right"),
            Annotation("3", "Structure viewer", "widget:show_nep_interface.struct_widget", "top-left"),
            Annotation("4", "Search and selection", "widget:show_nep_interface.search_lineEdit", "right"),
        ),
    ),
    "training_set_audit_overview": ScenarioSpec(
        name="training_set_audit_overview",
        title="Training Set Audit overview",
        runner="training_set_audit_overview",
        output=DEFAULT_OUTPUT_DIR / "training_set_audit_overview.png",
        window_size=(1440, 860),
        description="Audit overview showing the conclusion, dataset summary, review queue, and report export.",
        annotations=(
            Annotation("1", "先读当前结论", "widget:summary_panel", "right"),
            Annotation("2", "确认数据概况", "widget:metric_band", "top-right"),
            Annotation("3", "按建议顺序复核", "widget:next_actions_panel", "top-left"),
            Annotation("4", "导出分层报告", "widget:export_report_button", "left"),
        ),
    ),
    "training_set_audit_structure_map": ScenarioSpec(
        name="training_set_audit_structure_map",
        title="Phase evidence on the composition map",
        runner="training_set_audit_structure_map",
        output=DEFAULT_OUTPUT_DIR / "training_set_audit_structure_map.png",
        window_size=(1440, 860),
        description="Composition map after on-demand structural-phase analysis.",
        annotations=(
            Annotation("1", "切换元素与证据层", "widget:composition_view_selector", "left"),
            Annotation("2", "相分布随成分变化", "widget:composition_chart", "top-left"),
            Annotation("3", "精确成分与结构数", "widget:composition_table", "top-right"),
            Annotation("4", "筛选相并回看结构", "widget:composition_phase_selector", "left"),
        ),
    ),
    "training_set_audit_magnetic_shares": ScenarioSpec(
        name="training_set_audit_magnetic_shares",
        title="Magnetic-type shares",
        runner="training_set_audit_magnetic_shares",
        output=DEFAULT_OUTPUT_DIR / "training_set_audit_magnetic_shares.png",
        window_size=(1440, 860),
        description="Frame-normalized magnetic-type shares in Advanced evidence.",
        annotations=(
            Annotation("1", "选择磁类型证据", "widget:dimension_list", "right"),
            Annotation("2", "切换三种占比关系", "widget:plot_selector", "left"),
            Annotation("3", "按结构帧查看占比", "widget:chart_widget", "top-right"),
            Annotation("4", "点击色块回看结构", "widget:chart_send_button", "left"),
        ),
    ),
    "show_nep_index_dialog": ScenarioSpec(
        name="show_nep_index_dialog",
        title="Select by Index dialog",
        runner="show_nep_index_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_index_dialog.png",
        description="Show NEP index-selection dialog.",
        options={"render_scale": 2},
    ),
    "show_nep_range_dialog": ScenarioSpec(
        name="show_nep_range_dialog",
        title="Select by Range dialog",
        runner="show_nep_range_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_range_dialog.png",
        description="Show NEP scatter-range selection dialog.",
    ),
    "show_nep_lattice_dialog": ScenarioSpec(
        name="show_nep_lattice_dialog",
        title="Select by Lattice dialog",
        runner="show_nep_lattice_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_lattice_dialog.png",
        description="Show NEP lattice-range selection dialog.",
    ),
    "show_nep_max_error_dialog": ScenarioSpec(
        name="show_nep_max_error_dialog",
        title="Find max error dialog",
        runner="show_nep_max_error_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_maxerr_dialog.png",
        description="Show NEP maximum-error count dialog.",
        options={"render_scale": 2},
    ),
    "show_nep_sparse_dialog": ScenarioSpec(
        name="show_nep_sparse_dialog",
        title="Sparse samples dialog",
        runner="show_nep_sparse_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_sparse_dialog.png",
        description="Show NEP sparse sampling dialog.",
        options={"render_scale": 2},
    ),
    "show_nep_force_dialog": ScenarioSpec(
        name="show_nep_force_dialog",
        title="Check net force dialog",
        runner="show_nep_force_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_force_dialog.png",
        description="Show NEP net-force threshold dialog.",
        options={"render_scale": 2},
    ),
    "show_nep_edit_info_dialog": ScenarioSpec(
        name="show_nep_edit_info_dialog",
        title="Edit info dialog",
        runner="show_nep_edit_info_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_editinfo_dialog.png",
        description="Show NEP structure-info editing dialog.",
        options={"render_scale": 2},
    ),
    "show_nep_shift_dialog": ScenarioSpec(
        name="show_nep_shift_dialog",
        title="Energy baseline shift dialog",
        runner="show_nep_shift_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_shift_dialog.png",
        description="Show NEP energy-baseline shift dialog.",
        options={"render_scale": 2},
    ),
    "energy_baseline_shift_entry": ScenarioSpec(
        name="energy_baseline_shift_entry",
        title="Energy baseline shift entry",
        runner="energy_baseline_shift_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "energy_baseline_shift_entry.png",
        window_size=(1600, 850),
        description="Show the energy plot and the energy-baseline shift action.",
        annotations=(
            Annotation(
                "1",
                "Confirm a large constant energy baseline",
                (90, 150, 760, 470),
                "bottom-left",
            ),
            Annotation(
                "2",
                "Open energy baseline shift",
                "widget:energy_baseline_shift_button",
                "right",
            ),
        ),
    ),
    "energy_baseline_shift_result": ScenarioSpec(
        name="energy_baseline_shift_result",
        title="Energy plot after baseline shift",
        runner="energy_baseline_shift_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "energy_baseline_shift_result.png",
        window_size=(1600, 850),
        description="Show the energy plot after removing the tutorial's known atomic baseline.",
        annotations=(
            Annotation(
                "1",
                "The large constant baseline is removed",
                (90, 150, 760, 470),
                "bottom-left",
            ),
        ),
    ),
    "fps_sampling_entry": ScenarioSpec(
        name="fps_sampling_entry",
        title="Farthest point sampling",
        runner="fps_sampling_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "fps_sampling_entry.png",
        window_size=(1600, 850),
        description="Show the descriptor plot and farthest-point sampling action.",
        annotations=(
            Annotation(
                "1",
                "Inspect diversity in descriptor space",
                (90, 150, 760, 470),
                "bottom-left",
            ),
            Annotation(
                "2",
                "Open farthest point sampling",
                "widget:sparse_samples_button",
                "right",
            ),
        ),
    ),
    "fps_sampling_result": ScenarioSpec(
        name="fps_sampling_result",
        title="FPS representatives",
        runner="fps_sampling_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "fps_sampling_result.png",
        window_size=(1600, 850),
        description="Show eight structures selected by the tutorial FPS settings.",
        annotations=(
            Annotation(
                "1",
                "Eight representatives are selected",
                (90, 150, 760, 470),
                "bottom-left",
            ),
        ),
    ),
    "max_error_review_entry": ScenarioSpec(
        name="max_error_review_entry",
        title="Maximum-error review",
        runner="max_error_review_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "max_error_review_entry.png",
        window_size=(1600, 850),
        description="Show the active energy plot and maximum-error action.",
        annotations=(
            Annotation("1", "Activate the energy plot", (90, 150, 760, 470), "bottom-left"),
            Annotation(
                "2",
                "Find maximum-error structures",
                "widget:find_max_error_button",
                "right",
            ),
        ),
    ),
    "max_error_review_result": ScenarioSpec(
        name="max_error_review_result",
        title="Selected structures need review",
        runner="max_error_review_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "max_error_review_result.png",
        window_size=(1600, 850),
        description="Show five structures selected by energy error for manual review.",
        annotations=(
            Annotation("1", "Selected structures need review", (90, 150, 760, 470), "bottom-left"),
            Annotation("2", "Inspect structure and Config_type", (860, 95, 725, 700), "top-left"),
        ),
    ),
    "select_by_index_entry": ScenarioSpec(
        name="select_by_index_entry",
        title="Select structures by index",
        runner="select_by_index_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "select_by_index_entry.png",
        window_size=(1600, 850),
        description="Show index selection on a loaded NEP dataset.",
        annotations=(
            Annotation("1", "Open index selection", "widget:select_by_index_button", "right"),
        ),
    ),
    "select_by_index_result": ScenarioSpec(
        name="select_by_index_result",
        title="Select structures by index",
        runner="select_by_index_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "select_by_index_result.png",
        window_size=(1600, 850),
        description="Show the five structures resolved from the tutorial index expression.",
        annotations=(
            Annotation("1", "Five requested structures are selected", (90, 150, 760, 470), "bottom-left"),
        ),
    ),
    "suspicious_structure_scan_entry": ScenarioSpec(
        name="suspicious_structure_scan_entry",
        title="Structure quality checks",
        runner="suspicious_structure_scan_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "structure_quality_checks.png",
        window_size=(1600, 850),
        description="Show the distance and net-force quality checks.",
        annotations=(
            Annotation("1", "Scan suspicious distances", "widget:find_non_physical_button", "right"),
            Annotation("2", "Scan non-zero net forces", "widget:check_net_force_button", "right"),
        ),
    ),
    "suspicious_structure_scan_result": ScenarioSpec(
        name="suspicious_structure_scan_result",
        title="Structure quality checks",
        runner="suspicious_structure_scan_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "structure_quality_result.png",
        window_size=(1600, 850),
        description="Show real selections from deliberately corrupted geometry and force labels.",
        annotations=(
            Annotation(
                "1",
                "Two deliberately corrupted frames are selected",
                (90, 150, 760, 470),
                "bottom-left",
            ),
            Annotation("2", "Inspect structure and Config_type", (860, 95, 725, 700), "top-left"),
        ),
    ),
    "edit_metadata_entry": ScenarioSpec(
        name="edit_metadata_entry",
        title="Edit selected metadata",
        runner="edit_metadata_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "edit_metadata_entry.png",
        window_size=(1600, 850),
        description="Show metadata editing after selecting three structures.",
        annotations=(
            Annotation("1", "Select the structures to edit", (90, 150, 760, 470), "bottom-left"),
            Annotation("2", "Open the metadata editor", "widget:edit_info_button", "right"),
        ),
    ),
    "edit_metadata_result": ScenarioSpec(
        name="edit_metadata_result",
        title="Edit selected metadata",
        runner="edit_metadata_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "edit_metadata_result.png",
        window_size=(1600, 850),
        description="Show the updated Config_type on the selected structures.",
        annotations=(
            Annotation(
                "1",
                "Config_type is updated for the selection",
                "widget:show_nep_interface.struct_info_widget.config_text",
                "right",
            ),
        ),
    ),
    "dataset_summary_result": ScenarioSpec(
        name="dataset_summary_result",
        title="Dataset summary from real data",
        runner="dataset_summary_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "dataset_summary.png",
        window_size=(1600, 900),
        description="Show a summary computed from the tracked 25-structure demo dataset.",
    ),
    "distribution_analysis_result": ScenarioSpec(
        name="distribution_analysis_result",
        title="Force distribution from real data",
        runner="distribution_analysis_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "force_distribution.png",
        window_size=(1600, 900),
        description="Show a real force-component distribution analysis.",
    ),
    "dft_d3_entry": ScenarioSpec(
        name="dft_d3_entry",
        title="DFT-D3 correction entry",
        runner="dft_d3_entry",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "dft_d3_entry.png",
        window_size=(1600, 850),
        description="Show the reference energy plot and DFT-D3 action.",
        annotations=(
            Annotation(
                "1",
                "Confirm the current reference labels",
                (90, 150, 760, 470),
                "bottom-left",
            ),
            Annotation("2", "Open DFT-D3 correction", "widget:dft_d3_button", "right"),
        ),
    ),
    "dft_d3_result": ScenarioSpec(
        name="dft_d3_result",
        title="Energy plot after adding DFT-D3",
        runner="dft_d3_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "dft_d3_result.png",
        window_size=(1600, 850),
        description="Show the real demo labels after adding a PBE DFT-D3 correction.",
        annotations=(
            Annotation(
                "1",
                "Reference labels now include DFT-D3",
                (90, 150, 760, 470),
                "bottom-left",
            ),
        ),
    ),
    "show_nep_dftd3_dialog": ScenarioSpec(
        name="show_nep_dftd3_dialog",
        title="DFT D3 dialog",
        runner="show_nep_dftd3_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_dftd3_dialog.png",
        description="Show NEP DFT-D3 correction dialog.",
        options={"render_scale": 2},
    ),
    "show_nep_summary_dialog": ScenarioSpec(
        name="show_nep_summary_dialog",
        title="Dataset summary dialog",
        runner="show_nep_summary_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_summary_dialog.png",
        description="Show NEP dataset summary dialog.",
    ),
    "show_nep_distribution_dialog": ScenarioSpec(
        name="show_nep_distribution_dialog",
        title="Distribution inspector dialog",
        runner="show_nep_distribution_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_dist_dialog.png",
        description="Show NEP distribution inspector dialog.",
    ),
    "show_nep_arrow_dialog": ScenarioSpec(
        name="show_nep_arrow_dialog",
        title="Show arrows dialog",
        runner="show_nep_arrow_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "s_arrow_dialog.png",
        description="Show NEP vector-arrow configuration dialog.",
    ),
    "show_nep_export_format_dialog": ScenarioSpec(
        name="show_nep_export_format_dialog",
        title="Export format dialog",
        runner="show_nep_export_format_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "s_export_format.png",
        description="Show NEP export-format dialog.",
    ),
    "show_nep_drop_bad_dialog": ScenarioSpec(
        name="show_nep_drop_bad_dialog",
        title="Drop bad confirmation dialog",
        runner="show_nep_drop_bad_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "s_dropbad_confirm.png",
        description="Show NEP reject-deletion confirmation dialog.",
    ),
    "make_data_empty": ScenarioSpec(
        name="make_data_empty",
        title="Make Data workspace",
        runner="make_data_empty",
        output=DEFAULT_OUTPUT_DIR / "make_data_empty.png",
        description="Empty Make Data workflow page with the add-card entry point annotated.",
        annotations=(
            Annotation("1", "Open input structures", "widget:open_dir_button", "right"),
            Annotation("2", "Add new card", "widget:make_data_interface.setting_group.new_card_button", "right"),
            Annotation("3", "Run selected cards", (292, 72, 26, 26), "right"),
            Annotation("4", "Workflow workspace", "widget:make_data_interface.workspace_card_widget", "top-left"),
        ),
    ),
    "make_data_lattice_strain": ScenarioSpec(
        name="make_data_lattice_strain",
        title="Make Data lattice strain quickstart",
        runner="make_data_lattice_strain",
        output=DEFAULT_OUTPUT_DIR / "make_data_lattice_strain.png",
        description="Quickstart scene with a Lattice Strain card configured for small axial strain.",
        annotations=(
            Annotation("1", "Add Lattice Strain card", "widget:make_data_interface.setting_group.new_card_button", "right"),
            Annotation("2", "Set strain range", "widget:make_data_interface.workspace_card_widget.cards.0", "right"),
            Annotation("3", "Run selected cards", (292, 72, 26, 26), "right"),
            Annotation("4", "Export generated structures", "widget:make_data_interface.workspace_card_widget.cards.0.export_button", "right"),
        ),
    ),
    "make_data_card_system_controls": ScenarioSpec(
        name="make_data_card_system_controls",
        title="Shared card controls",
        runner="make_data_card_system_controls",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "card_system_controls.png",
        window_size=(1500, 820),
        description="Show the controls shared by Make Dataset cards.",
        annotations=(
            Annotation("1", "Enable or skip this card", "widget:card_workflow_supercell.state_checkbox", "right"),
            Annotation("2", "Collapse or expand settings", "widget:card_workflow_supercell.collapse_button", "right"),
            Annotation("3", "Open this card's manual", "widget:card_workflow_supercell.doc_button", "right"),
            Annotation("4", "Show card provenance", "widget:card_workflow_supercell.info_button", "right"),
            Annotation("5", "Copy this card as JSON", "widget:card_workflow_supercell.copy_json_button", "right"),
            Annotation("6", "View this card's output", "widget:card_workflow_supercell.view_output_button", "right"),
            Annotation("7", "Export this card's output", "widget:card_workflow_supercell.export_button", "right"),
            Annotation("8", "Remove this card", "widget:card_workflow_supercell.close_button", "right"),
        ),
    ),
    "make_data_card_system_workflow": ScenarioSpec(
        name="make_data_card_system_workflow",
        title="A complete card workflow",
        runner="make_data_card_system_workflow",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "card_system_workflow.png",
        window_size=(1800, 1000),
        description="Show an ordered workflow with a two-branch Card Group.",
        annotations=(
            Annotation("1", "Add or search for cards", "widget:make_data_interface.setting_group.new_card_button", "right"),
            Annotation("2", "Copy the whole workflow as JSON", "widget:make_data_copy_button", "right"),
            Annotation("3", "Top-level cards run in order", (48, 106, 1750, 70), "top-right"),
            Annotation("4", "Grouped cards branch from one input", "widget:card_workflow_group", "top-left"),
            Annotation("5", "Run all enabled cards", "widget:make_data_run_button", "right"),
        ),
    ),
    "make_data_card_system_result": ScenarioSpec(
        name="make_data_card_system_result",
        title="Verified workflow result",
        runner="make_data_card_system_result",
        output=DEFAULT_OUTPUT_DIR / "tutorials" / "card_system_result.png",
        window_size=(1800, 1000),
        description="Show verified intermediate and final output counts for the card workflow.",
        annotations=(
            Annotation("1", "Two branches produce four structures", "widget:card_workflow_group", "top-left"),
            Annotation("2", "The final filter keeps eight structures", "widget:card_workflow_filter.status_label", "right"),
            Annotation("3", "View all enabled card outputs", "widget:make_data_interface.setting_group.view_output_button", "right"),
            Annotation("4", "Inspect the final card output", "widget:card_workflow_filter.view_output_button", "right"),
            Annotation("5", "Export the final card output", "widget:card_workflow_filter.export_button", "right"),
        ),
    ),
}
