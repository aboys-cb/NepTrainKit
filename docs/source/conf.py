# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:

    sys.path.insert(0, str(SRC_DIR))
project = 'NepTrainKit'
copyright = '2024, NepTrain Team'
author = 'NepTrain Team'
release = '1.4.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
html_show_sourcelink = False
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx_copybutton',
    'myst_parser',
]
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
}
autodoc_typehints = 'description'

API_PACKAGE_ROOTS = ['NepTrainKit']
SKIP_PACKAGE_PREFIXES = ('NepTrainKit.ui',)
autodoc_mock_imports = [
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'qfluentwidgets',
    'qframelesswindow',
    'vispy',
    'pyqtgraph',
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    'amsmath',
    'attrs_inline',
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    # 'linkify',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

templates_path = ['_templates']
exclude_patterns = [
    'module/nep-dataset-display-content.md',
    'module/training-set-assessment-content.md',
]
locale_dirs = ['locale/']
gettext_compact = False
gettext_uuid = True
gettext_additional_targets = {'image', 'literal-block'}
_rtd_language = os.environ.get('READTHEDOCS_LANGUAGE', '').lower()
language = {
    'zh-cn': 'zh_CN',
    'zh_cn': 'zh_CN',
}.get(_rtd_language, _rtd_language or 'zh_CN')


html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # Keep the global sidebar as a document tree.  Page-local headings belong
    # in the page's own contents block; mixing both makes long reference pages
    # unreadable in the navigation.
    'collapse_navigation': True,
    'navigation_depth': 5,
    'includehidden': True,
    'titles_only': True,
}
html_static_path = ['_static']
html_context = {
    'author_name': author,
}
html_css_files = [
    'css/custom.css',  # specify your custom CSS file here
]
html_js_files = [
    'js/partial-navigation.js',
]

copybutton_prompt_text = r'^\s*(>>> |\.\.\. |\$ |PS> )'
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Some autodoc-parsed docstrings use ``|ΣF|`` style markers.
# Define the substitution globally so Sphinx -W builds stay strict-clean.
rst_prolog = """
.. |ΣF| replace:: ΣF
"""

_BILINGUAL_SCREENSHOTS = (
    'make_data_empty.png',
    'make_data_lattice_strain.png',
    'show_nep_overview.png',
    'training_set_audit_overview.png',
    'training_set_audit_structure_map.png',
    'training_set_audit_magnetic_shares.png',
    'g_index_dialog.png',
    'g_range_dialog.png',
    'g_lattice_dialog.png',
    'g_maxerr_dialog.png',
    'g_sparse_dialog.png',
    'g_force_dialog.png',
    'g_editinfo_dialog.png',
    'g_shift_dialog.png',
    'g_structure_filter.png',
    'g_structure_filter_presets.png',
    'energy_baseline_shift_entry.png',
    'energy_baseline_shift_result.png',
    'fps_sampling_entry.png',
    'fps_sampling_result.png',
    'max_error_review_entry.png',
    'max_error_review_result.png',
    'dft_d3_entry.png',
    'dft_d3_result.png',
    'select_by_index_entry.png',
    'select_by_index_result.png',
    'structure_quality_checks.png',
    'structure_quality_result.png',
    'edit_metadata_entry.png',
    'edit_metadata_result.png',
    'force_distribution.png',
    'card_system_controls.png',
    'card_system_workflow.png',
    'card_system_result.png',
    'g_dftd3_dialog.png',
    'g_dist_dialog.png',
    's_arrow_dialog.png',
    's_export_format.png',
    's_dropbad_confirm.png',
)


def _use_english_screenshots(app, docname, source):
    if app.config.language != 'en':
        return

    # MyST keeps numbered section headings as literal source text in a few
    # tutorial pages, so they bypass the normal gettext title lookup.  Replace
    # those headings at source-read time to keep the English build entirely
    # English while leaving the Chinese source and build unchanged.
    heading_translations = {
        '1. 安装': '1. Install NepTrainKit',
        '2. 打开 生成数据集': '2. Open Make Dataset',
        '3. 生成一组候选结构': '3. Generate a candidate set',
        '4. 检查候选结构': '4. Inspect the candidates',
        '5. 进入 DFT 和训练': '5. Continue to DFT and training',
        '1. 打开训练结果': '1. Open the training results',
        '2. 先看误差集中在哪里': '2. Locate the error concentration',
        '3. 进入训练集评估': '3. Open Training Set Audit',
        '4. 给每类问题选择处理方向': '4. Choose a response for each problem type',
        '5. 导出可追溯的下一轮输入': '5. Export traceable inputs for the next round',
        '1. 先保存原始数据': '1. Save the original data first',
        '2. 泛函必须与原计算一致': '2. Match the original functional',
        '3. 看标签是否按预期改变': '3. Check the updated labels',
        '1. 先激活要检查的图': '1. Activate the plot to review',
        '2. 输入要复核的结构数': '2. Enter the number of structures',
        '3. 逐个判断误差来源': '3. Determine the error source',
        '1. 打开描述符图': '1. Open the descriptor plot',
        '2. 第一次这样设置': '2. Use this initial setup',
        '3. 导出选中的代表结构': '3. Export the selected representatives',
        '1. 先看能量图': '1. Inspect the energy plot',
        '2. 按这个例子设置': '2. Use the example settings',
        '3. 看结果是否正确': '3. Verify the result',
        '1. 打开结构和模型': '1. Open the structures and model',
        '2. 逐帧查看结构': '2. Inspect structures frame by frame',
        '3. 从最大误差回到原结构': '3. Return to the original structure',
        '4. 用 FPS 选择代表结构': '4. Select representatives with FPS',
        '5. 在图上直接框选或反选': '5. Select or invert directly on the plot',
        '6. 按 Config_type 组合筛选': '6. Filter by Config_type groups',
        '7. 只导出确认过的子集': '7. Export only the reviewed subset',
        '1. 界面总览': '1. Interface overview',
        '5. 导入导出与 NEP 模型切换': '5. Import/export and NEP model switching',
        '6. 结构筛选栏': '6. Structure filter bar',
        '7. 状态与常见提示': '7. Status and common messages',
        '6. 按 `Config_type` 组合筛选': '6. Filter by `Config_type` groups',
        '训练集评估概览页，标出当前结论、数据概况、建议复核顺序和 HTML 报告导出入口': 'Training Set Audit overview with conclusions, data summary, review order, and HTML report export',
        '训练集评估的磁类型占比视图，显示图表切换、结构帧占比、颜色图例和结构回选入口': 'Training Set Audit magnetic-type shares with chart controls, frame shares, legend, and structure selection',
        '训练集评估的组分地图，标出证据层切换、相分布柱图、精确组分表和结构回选控件': 'Training Set Audit composition map with evidence controls, phase bars, exact composition table, and structure selection',
    }
    for chinese, english in heading_translations.items():
        source[0] = source[0].replace(f'## {chinese}', f'## {english}')
        source[0] = source[0].replace(chinese, english)

    for filename in _BILINGUAL_SCREENSHOTS:
        english_filename = f'{Path(filename).stem}_en{Path(filename).suffix}'
        source[0] = source[0].replace(filename, english_filename)

    source[0] = source[0].replace(
        '../_static/image/example/display/main.png',
        '../_static/image/generated/show_nep_overview_en.png',
    )


def _replace_english_fallback_text(app, exception):
    """Replace headings/alt text from included legacy pages in English HTML.

    A few large included Markdown files are intentionally excluded from the
    gettext catalogs.  Their section titles and image alt text therefore
    remain Chinese even though the surrounding paragraphs are translated.
    Keep the source shared, but make the English output self-contained.
    """
    if exception is not None or app.config.language != 'en':
        return
    replacements = {
        '1. 界面总览': '1. Interface overview',
        '5. 导入导出与 NEP 模型切换': '5. Import/export and NEP model switching',
        '6. 结构筛选栏': '6. Structure filter bar',
        '7. 状态与常见提示': '7. Status and common messages',
        '训练集评估概览页，标出当前结论、数据概况、建议复核顺序和 HTML 报告导出入口': 'Training Set Audit overview with conclusions, data summary, review order, and HTML report export',
        '训练集评估的磁类型占比视图，显示图表切换、结构帧占比、颜色图例和结构回选入口': 'Training Set Audit magnetic-type shares with chart controls, frame shares, legend, and structure selection',
        '训练集评估的组分地图，标出证据层切换、相分布柱图、精确组分表和结构回选控件': 'Training Set Audit composition map with evidence controls, phase bars, exact composition table, and structure selection',
    }
    for path in app.outdir.rglob('*.html'):
        text = path.read_text(encoding='utf-8')
        updated = text
        for chinese, english in replacements.items():
            updated = updated.replace(chinese, english)
        if updated != text:
            path.write_text(updated, encoding='utf-8')


def setup(app):
    app.connect('source-read', _use_english_screenshots)
    app.connect('build-finished', _replace_english_fallback_text)
