# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
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
# locale_dirs = ['docs/locales']  # directory for translation files
language = 'en'


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_context = {
    'author_name': author,
}
html_css_files = [
    'css/custom.css',  # specify your custom CSS file here
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

