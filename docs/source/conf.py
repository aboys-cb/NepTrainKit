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
html_static_path = ['_static']
html_context = {
    'author_name': author,
}
html_css_files = [
    'css/custom.css',  # specify your custom CSS file here
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
    'dataset_summary.png',
    'force_distribution.png',
    'card_system_controls.png',
    'card_system_workflow.png',
    'card_system_result.png',
    'g_dftd3_dialog.png',
    'g_summary_dialog.png',
    'g_dist_dialog.png',
    's_arrow_dialog.png',
    's_export_format.png',
    's_dropbad_confirm.png',
)


def _use_english_screenshots(app, docname, source):
    if app.config.language != 'en':
        return

    for filename in _BILINGUAL_SCREENSHOTS:
        english_filename = f'{Path(filename).stem}_en{Path(filename).suffix}'
        source[0] = source[0].replace(filename, english_filename)

    source[0] = source[0].replace(
        '../_static/image/example/display/main.png',
        '../_static/image/generated/show_nep_overview_en.png',
    )


def setup(app):
    app.connect('source-read', _use_english_screenshots)
