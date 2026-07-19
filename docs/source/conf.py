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

REVIEWED_ENGLISH_PAGES = {
    'index',
    'quickstart',
    'module/index',
    'module/NEP-dataset-display',
    'module/training-set-assessment',
}


def _add_translation_status(app, pagename, templatename, context, doctree):
    if (
        doctree is None
        or app.config.language != 'en'
        or pagename in REVIEWED_ENGLISH_PAGES
    ):
        return

    notice = (
        '<aside class="docs-translation-status" role="note">'
        '<strong>English review in progress.</strong> '
        'This page currently falls back to the Chinese source so that no technical detail is hidden. '
        'The five core entry pages are already available in reviewed English.'
        '</aside>'
    )
    context['body'] = notice + context['body']


def setup(app):
    app.connect('html-page-context', _add_translation_status)
