# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'NepTrainKit'
copyright = '2024, NepTrain Team'
author="NepTrain Team"
release = '1.4.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
html_show_sourcelink = False
extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.mathjax',

"sphinx_design",
    "myst_parser"
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
# locale_dirs = ['docs/locales']  # 语言文件目录
language = 'en'


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_context = {
    "author_name": author,
}
html_css_files = [
    'css/custom.css',  # 你可以在这里指定你的自定义 CSS 文件
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}