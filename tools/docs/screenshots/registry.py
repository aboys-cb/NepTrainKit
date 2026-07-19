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
    "Make Data workspace": "Make Data 工作区",
    "Open input structures": "打开输入结构",
    "Add new card": "添加卡片",
    "Run selected cards": "运行选中卡片",
    "Workflow workspace": "流程工作区",
    "Make Data lattice strain quickstart": "Make Data 晶格应变快速上手",
    "Add Lattice Strain card": "添加 Lattice Strain 卡片",
    "Set strain range": "设置应变范围",
    "Export generated structures": "导出生成结构",
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
    "show_nep_index_dialog": ScenarioSpec(
        name="show_nep_index_dialog",
        title="Select by Index dialog",
        runner="show_nep_index_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_index_dialog.png",
        description="Show NEP index-selection dialog.",
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
    ),
    "show_nep_sparse_dialog": ScenarioSpec(
        name="show_nep_sparse_dialog",
        title="Sparse samples dialog",
        runner="show_nep_sparse_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_sparse_dialog.png",
        description="Show NEP sparse sampling dialog.",
    ),
    "show_nep_force_dialog": ScenarioSpec(
        name="show_nep_force_dialog",
        title="Check net force dialog",
        runner="show_nep_force_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_force_dialog.png",
        description="Show NEP net-force threshold dialog.",
    ),
    "show_nep_edit_info_dialog": ScenarioSpec(
        name="show_nep_edit_info_dialog",
        title="Edit info dialog",
        runner="show_nep_edit_info_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_editinfo_dialog.png",
        description="Show NEP structure-info editing dialog.",
    ),
    "show_nep_shift_dialog": ScenarioSpec(
        name="show_nep_shift_dialog",
        title="Energy baseline shift dialog",
        runner="show_nep_shift_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_shift_dialog.png",
        description="Show NEP energy-baseline shift dialog.",
    ),
    "show_nep_dftd3_dialog": ScenarioSpec(
        name="show_nep_dftd3_dialog",
        title="DFT D3 dialog",
        runner="show_nep_dftd3_dialog",
        output=DEFAULT_OUTPUT_DIR / "show_nep_reference" / "g_dftd3_dialog.png",
        description="Show NEP DFT-D3 correction dialog.",
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
}
